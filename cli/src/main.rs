//! The `neura` command-line driver.
//!
//! Every command goes through the same real pipeline used by the libraries:
//! lex/parse/type-check/ownership-check -> EIR -> NAOE -> execution.
//! Nothing here shells out to a Python runtime.

use neura_compiler::diagnostics::render_all;
use neura_compiler::lowering::hash_source;
use neura_compiler::CompileOutput;
use neura_ir::eir::{CompilePolicies, DevicePolicy, OptimizeMode, PrecisionPolicy};
use neura_ir::mlir;
use neura_ir::tir::TirGraph;
use neura_optimizer::OptimizationReport;
use neura_runtime::support::CompilationCache;
use neura_runtime::{run, VmOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const VERSION: &str = "0.1.0";

const USAGE: &str = "\
neura 0.1.0 — compiler and runtime for the Neura language

usage: neura <command> <file.neura> [options]

commands:
  check             parse, type-check and ownership-check a program
  build             compile and write a .neurabc artifact
  run               compile, optimise and execute main
  inspect           print the typed EIR program and inferred types
  graph             print the TIR graph as MLIR textual form
  profile           run with per-operation timing and memory accounting
  optimize-report   print the NAOE optimisation report
  fmt               rewrite a file in canonical form (use --check to verify)
  version           print the compiler version

options:
  --opt=<mode>        auto (default) | aggressive | debug | off
  --device=<dev>      auto (default) | cpu | cuda | rocm
  --precision=<p>     strict (default) | mixed_bf16 | mixed_fp16
  --output=<path>     artifact path for `build`
  --json              emit machine-readable JSON where supported
  --check             `fmt`: do not write, exit 1 when reformatting is needed
  --trace             `run`/`profile`: print every executed instruction
  --quiet             `run`/`profile`: suppress program output
  --training          `run`/`profile`: run layers in training mode
";

#[derive(Default)]
struct Options {
    optimize: Option<OptimizeMode>,
    device: Option<DevicePolicy>,
    precision: Option<PrecisionPolicy>,
    json: bool,
    check: bool,
    trace: bool,
    quiet: bool,
    training: bool,
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {}", message);
            ExitCode::from(2)
        }
    }
}

fn dispatch(args: &[String]) -> Result<ExitCode, String> {
    let Some(command) = args.first() else {
        print!("{}", USAGE);
        return Ok(ExitCode::SUCCESS);
    };
    if command == "help" || command == "--help" || command == "-h" {
        print!("{}", USAGE);
        return Ok(ExitCode::SUCCESS);
    }
    if command == "version" || command == "--version" || command == "-V" {
        println!("neura {}", VERSION);
        return Ok(ExitCode::SUCCESS);
    }

    let mut file: Option<String> = None;
    let mut options = Options::default();
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        if let Some(value) = arg.strip_prefix("--opt=") {
            options.optimize = Some(
                OptimizeMode::parse(value).ok_or_else(|| format!("unknown optimisation mode `{}`", value))?,
            );
        } else if let Some(value) = arg.strip_prefix("--device=") {
            options.device = Some(
                DevicePolicy::parse(value).ok_or_else(|| format!("unknown device `{}`", value))?,
            );
        } else if let Some(value) = arg.strip_prefix("--precision=") {
            options.precision = Some(
                PrecisionPolicy::parse(value).ok_or_else(|| format!("unknown precision `{}`", value))?,
            );
        } else if let Some(value) = arg.strip_prefix("--output=") {
            options.output = Some(PathBuf::from(value));
        } else if arg == "--json" {
            options.json = true;
        } else if arg == "--check" {
            options.check = true;
        } else if arg == "--trace" {
            options.trace = true;
        } else if arg == "--quiet" {
            options.quiet = true;
        } else if arg == "--training" {
            options.training = true;
        } else if arg.starts_with('-') {
            return Err(format!("unknown option `{}`", arg));
        } else if file.is_none() {
            file = Some(arg.clone());
        } else {
            return Err(format!("unexpected argument `{}`", arg));
        }
    }

    let Some(file) = file else {
        return Err(format!("`{}` requires a source file\n\n{}", command, USAGE));
    };

    match command.as_str() {
        "check" => cmd_check(&file, &options),
        "build" => cmd_build(&file, &options),
        "run" => cmd_run(&file, &options),
        "inspect" => cmd_inspect(&file, &options),
        "graph" => cmd_graph(&file, &options),
        "profile" => cmd_profile(&file, &options),
        "optimize-report" => cmd_optimize_report(&file, &options),
        "fmt" => cmd_fmt(&file, &options),
        other => Err(format!("unknown command `{}`\n\n{}", other, USAGE)),
    }
}

// ---------------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------------

/// A `.neurabc` artifact embeds the source it was built from together with the
/// policies and a checked hash. Building is therefore reproducible and loading
/// an artifact cannot silently diverge from the source it names.
struct Artifact {
    source: String,
    policies: CompilePolicies,
}

fn read_input(file: &str) -> Result<Artifact, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("cannot read {}: {}", file, e))?;
    if Path::new(file).extension().and_then(|e| e.to_str()) == Some("neurabc") {
        parse_artifact(&text)
    } else {
        Ok(Artifact { source: text, policies: CompilePolicies::default() })
    }
}

fn parse_artifact(text: &str) -> Result<Artifact, String> {
    let (header, source) = text
        .split_once("\n---\n")
        .ok_or_else(|| "malformed artifact: missing `---` separator".to_string())?;
    let mut hash = None;
    let mut policies = CompilePolicies::default();
    for line in header.lines() {
        let mut parts = line.splitn(2, ' ');
        let key = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");
        match key {
            "neurabc" => {
                if value != "1" {
                    return Err(format!("unsupported artifact version `{}`", value));
                }
            }
            "source-hash" => hash = Some(value.to_string()),
            "optimize" => {
                policies.optimize =
                    OptimizeMode::parse(value).ok_or_else(|| format!("bad artifact mode `{}`", value))?
            }
            "device" => {
                policies.device =
                    DevicePolicy::parse(value).ok_or_else(|| format!("bad artifact device `{}`", value))?
            }
            "precision" => {
                policies.precision =
                    PrecisionPolicy::parse(value).ok_or_else(|| format!("bad precision `{}`", value))?
            }
            _ => {}
        }
    }
    let expected = hash.ok_or_else(|| "malformed artifact: no source-hash".to_string())?;
    let actual = format!("{:016x}", hash_source(source));
    if expected != actual {
        return Err(format!(
            "artifact hash mismatch: header says {}, source hashes to {}",
            expected, actual
        ));
    }
    Ok(Artifact { source: source.to_string(), policies })
}

fn apply_policies(mut policies: CompilePolicies, options: &Options) -> CompilePolicies {
    if let Some(o) = options.optimize {
        policies.optimize = o;
    }
    if let Some(d) = options.device {
        policies.device = d;
    }
    if let Some(p) = options.precision {
        policies.precision = p;
    }
    policies
}

fn compile_artifact(artifact: &Artifact, options: &Options) -> Result<CompileOutput, (Vec<String>, String)> {
    let policies = apply_policies(artifact.policies, options);
    let out = neura_compiler::compile(&artifact.source, policies);
    if neura_compiler::has_errors(&out.diags) {
        let rendered = render_all(&out.diags, &artifact.source);
        return Err((rendered.lines().map(|l| l.to_string()).collect(), artifact.source.clone()));
    }
    Ok(out)
}

fn emit_diagnostics(diags: &[neura_compiler::Diagnostic], source: &str) {
    let text = render_all(diags, source);
    if !text.trim().is_empty() {
        eprint!("{}", text);
        if !text.ends_with('\n') {
            eprintln!();
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn artifact_text(output: &CompileOutput, report: &OptimizationReport) -> String {
    format!(
        "neurabc 1\nsource-hash {:016x}\nprogram {}\noptimize {}\ndevice {}\nprecision {}\ninstructions {}\n---\n{}",
        output.eir.source_hash,
        output.eir.name,
        output.eir.policies.optimize.name(),
        output.eir.policies.device.name(),
        output.eir.policies.precision.name(),
        report.instructions_after,
        // The embedded source is the checked input; recompiling it reproduces
        // the artifact exactly.
        source_of(output),
    )
}

/// The compiler does not retain the original text, so `build` embeds the
/// formatted source produced from the parsed program. This keeps the artifact
/// self-contained and byte-stable.
fn source_of(output: &CompileOutput) -> String {
    neura_formatter::format_program(&output.program)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_check(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let policies = apply_policies(artifact.policies, options);
    let out = neura_compiler::compile(&artifact.source, policies);
    let errors = neura_compiler::has_errors(&out.diags);
    if options.json {
        let diags: Vec<String> = out.diags.iter().map(|d| d.to_json()).collect();
        println!(
            "{{\"status\": {}, \"file\": {}, \"diagnostics\": [{}]}}",
            json_escape(if errors { "error" } else { "ok" }),
            json_escape(file),
            diags.join(", ")
        );
    } else {
        emit_diagnostics(&out.diags, &artifact.source);
        if !errors {
            let fns = out.report.functions.len();
            let imports = out.report.imports.len();
            println!("ok: {} function(s), {} import(s), {} struct(s)", fns, imports, out.report.structs.len());
        }
    }
    Ok(if errors { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

fn cmd_build(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    let mut program = out.eir.clone();
    let report = neura_optimizer::optimize(&mut program);
    let text = artifact_text(&CompileOutput { eir: program.clone(), ..out }, &report);

    let destination = options
        .output
        .clone()
        .unwrap_or_else(|| Path::new(file).with_extension("neurabc"));
    fs::write(&destination, &text)
        .map_err(|e| format!("cannot write {}: {}", destination.display(), e))?;
    // Keep the compilation cache in sync with what was produced.
    let cache = CompilationCache::default_location();
    let cached = cache.store(program.source_hash, &text).ok();

    if options.json {
        println!(
            "{{\"status\": \"ok\", \"artifact\": {}, \"cache\": {}, \"instructions\": {}}}",
            json_escape(&destination.display().to_string()),
            match &cached {
                Some(p) => json_escape(&p.display().to_string()),
                None => "null".to_string(),
            },
            report.instructions_after
        );
    } else {
        println!("built {}", destination.display());
        println!("{}", report.to_text().lines().take(1).collect::<Vec<_>>().join(""));
        if let Some(p) = cached {
            println!("cached {}", p.display());
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_run(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    let mut program = out.eir;
    let report = neura_optimizer::optimize(&mut program);
    let vm = VmOptions {
        trace: options.trace,
        quiet: options.quiet || options.json,
        training: options.training,
        profile: false,
    };
    let outcome = match run(&program, vm) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("runtime error: {}", e);
            return Ok(ExitCode::from(1));
        }
    };
    if options.json {
        println!(
            "{{\"status\": \"ok\", \"steps\": {}, \"output\": [{}], \"peak_bytes\": {}}}",
            outcome.steps,
            outcome
                .output
                .iter()
                .map(|l| json_escape(l))
                .collect::<Vec<_>>()
                .join(", "),
            outcome.memory.peak_bytes
        );
    } else if options.trace {
        println!("-- {} instruction(s), {} optimisation rewrite(s)", outcome.steps, report.instructions_before.saturating_sub(report.instructions_after));
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_inspect(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    if options.json {
        let functions: Vec<String> = out
            .report
            .functions
            .iter()
            .map(|f| {
                let locals: Vec<String> = f
                    .locals
                    .iter()
                    .map(|(n, t)| format!("{{\"name\": {}, \"type\": {}}}", json_escape(n), json_escape(&t.name())))
                    .collect();
                format!(
                    "{{\"name\": {}, \"return\": {}, \"locals\": [{}]}}",
                    json_escape(&f.name),
                    json_escape(&f.ret.name()),
                    locals.join(", ")
                )
            })
            .collect();
        println!(
            "{{\"program\": {}, \"optimize\": {}, \"device\": {}, \"precision\": {}, \"functions\": [{}]}}",
            json_escape(&out.eir.name),
            json_escape(out.eir.policies.optimize.name()),
            json_escape(out.eir.policies.device.name()),
            json_escape(out.eir.policies.precision.name()),
            functions.join(", ")
        );
    } else {
        println!("program: {}", out.eir.name);
        println!(
            "policies: optimize={} device={} precision={}",
            out.eir.policies.optimize.name(),
            out.eir.policies.device.name(),
            out.eir.policies.precision.name()
        );
        for f in &out.report.functions {
            let params: Vec<String> = f.params.iter().map(|(n, t)| format!("{}: {}", n, t.name())).collect();
            println!("fn {}({}) -> {}", f.name, params.join(", "), f.ret.name());
            for (name, ty) in &f.locals {
                println!("    {}: {}", name, ty.name());
            }
        }
        println!();
        print!("{}", out.eir.disassemble());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_graph(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    let mut printed = 0;
    for f in &out.eir.functions {
        match TirGraph::from_eir(f) {
            Ok(graph) => {
                print!("{}", mlir::print_tir(&f.name, &graph));
                printed += 1;
            }
            Err(reason) => {
                if !options.json {
                    eprintln!("note: {} has no tensor graph ({})", f.name, reason);
                }
            }
        }
    }
    if options.json {
        println!("{{\"status\": \"ok\", \"graphs\": {}}}", printed);
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_profile(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    let mut program = out.eir;
    let report = neura_optimizer::optimize(&mut program);
    let vm = VmOptions {
        trace: options.trace,
        quiet: options.quiet || options.json,
        training: options.training,
        profile: true,
    };
    let outcome = match run(&program, vm) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("runtime error: {}", e);
            return Ok(ExitCode::from(1));
        }
    };
    if options.json {
        let functions: Vec<String> = outcome
            .report
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| json_escape(l.trim_end()))
            .collect();
        println!(
            "{{\"status\": \"ok\", \"steps\": {}, \"peak_bytes\": {}, \"allocations\": {}, \"pooled_reuses\": {}, \"profile\": [{}]}}",
            outcome.steps,
            outcome.memory.peak_bytes,
            outcome.memory.allocations,
            outcome.memory.pooled_reuses,
            functions.join(", ")
        );
    } else {
        print!("{}", outcome.report);
        println!();
        println!(
            "memory: {} allocation(s), peak {} bytes, {} pooled reuse(s)",
            outcome.memory.allocations, outcome.memory.peak_bytes, outcome.memory.pooled_reuses
        );
        println!(
            "optimisation: {} -> {} instructions",
            report.instructions_before, report.instructions_after
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_optimize_report(file: &str, options: &Options) -> Result<ExitCode, String> {
    let artifact = read_input(file)?;
    let out = match compile_artifact(&artifact, options) {
        Ok(out) => out,
        Err((rendered, _)) => {
            for line in rendered {
                eprintln!("{}", line);
            }
            return Ok(ExitCode::from(1));
        }
    };
    let mut program = out.eir;
    let report = neura_optimizer::optimize(&mut program);
    if options.json {
        print!("{}", report.to_json());
    } else {
        print!("{}", report.to_text());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_fmt(file: &str, options: &Options) -> Result<ExitCode, String> {
    let source = fs::read_to_string(file).map_err(|e| format!("cannot read {}: {}", file, e))?;
    let formatted = neura_formatter::format_source(&source).map_err(|e| e)?;
    if formatted == source {
        if !options.json {
            println!("{} is already formatted", file);
        }
        return Ok(ExitCode::SUCCESS);
    }
    if options.check {
        if options.json {
            println!("{{\"status\": \"needs-formatting\", \"file\": {}}}", json_escape(file));
        } else {
            println!("{} needs formatting", file);
        }
        return Ok(ExitCode::from(1));
    }
    fs::write(file, &formatted).map_err(|e| format!("cannot write {}: {}", file, e))?;
    if options.json {
        println!("{{\"status\": \"formatted\", \"file\": {}}}", json_escape(file));
    } else {
        println!("formatted {}", file);
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neura_compiler::diagnostics::Severity;

    fn artifact_round_trip(src: &str) -> String {
        let policies = CompilePolicies::default();
        let out = neura_compiler::compile(src, policies);
        assert!(!neura_compiler::has_errors(&out.diags), "{:?}", out.diags);
        let mut program = out.eir.clone();
        let report = neura_optimizer::optimize(&mut program);
        let out = CompileOutput { eir: program, ..out };
        artifact_text(&out, &report)
    }

    #[test]
    fn artifact_round_trips_through_the_parser() {
        let text = artifact_round_trip("fn main():\n    x = 2 + 3\n    print(x)\n");
        let artifact = parse_artifact(&text).unwrap();
        assert!(artifact.source.contains("print(x)"));
        assert_eq!(artifact.policies.optimize, OptimizeMode::Auto);
    }

    #[test]
    fn artifact_hash_mismatch_is_rejected() {
        let text = artifact_round_trip("fn main():\n    print(1)\n").replace("print(1)", "print(2)");
        assert!(parse_artifact(&text).is_err());
    }

    #[test]
    fn policies_are_overridden_by_flags() {
        let mut options = Options::default();
        options.optimize = Some(OptimizeMode::Off);
        options.device = Some(DevicePolicy::Cpu);
        let policies = apply_policies(CompilePolicies::default(), &options);
        assert_eq!(policies.optimize, OptimizeMode::Off);
        assert_eq!(policies.device, DevicePolicy::Cpu);
        assert_eq!(policies.precision, PrecisionPolicy::Strict);
    }

    #[test]
    fn json_escaping_matches_json_expectations() {
        assert_eq!(json_escape("a\"b\n"), "\"a\\\"b\\n\"");
    }

    #[test]
    fn diagnostics_are_errors_only_when_present() {
        let out = neura_compiler::compile("fn main(:\n", CompilePolicies::default());
        assert!(out.diags.iter().any(|d| d.severity == Severity::Error));
    }
}
