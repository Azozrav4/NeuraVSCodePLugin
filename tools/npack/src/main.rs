//! `npack` — the Neura package manager.
//!
//! Packages are described by an `npack.toml` manifest. `npack` resolves the
//! declared dependencies (local `path` dependencies and packages already
//! present in the offline cache), writes a deterministic `npack.lock`, and
//! builds/runs/tests the package through the same compiler and runtime the
//! `neura` CLI uses.
//!
//! There is no network access in this build. A version-only dependency that is
//! absent from the local cache is reported as such instead of being silently
//! skipped or fabricated.
//!
//! Known limitation (documented in the help text): the language's module
//! system currently covers the `neura.*` standard library only, so dependency
//! packages are resolved, locked and build-verified but their symbols cannot
//! yet be imported from `.neura` sources.

use neura_compiler::diagnostics::render_all;
use neura_compiler::CompileOutput;
use neura_ir::eir::CompilePolicies;
use neura_runtime::{run as run_program, VmOptions};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const VERSION: &str = "0.1.0";
const MANIFEST: &str = "npack.toml";
const LOCKFILE: &str = "npack.lock";

const USAGE: &str = "\
npack 0.1.0 — package manager for Neura

usage: npack <command> [directory] [options]

commands:
  init       create npack.toml and src/main.neura in a new package
  install    resolve dependencies and write npack.lock
  build      compile the package into target/<name>.neurabc
  run        build and execute the package entry point
  test       compile and run every tests/*.neura file
  version    print the npack version

options:
  --quiet    suppress informational output
  --opt=<m>  auto (default) | aggressive | debug | off

manifest (npack.toml):
  [package]
  name = \"hello\"
  version = \"0.1.0\"
  entry = \"src/main.neura\"

  [dependencies]
  local = { path = \"../local\" }
  pinned = \"^0.1.0\"      # resolved from the offline cache only

limitation: dependency symbols cannot be imported yet; the compiler's module
system currently covers the neura.* standard library only.
";

fn main() -> ExitCode {
    match dispatch(std::env::args().skip(1).collect::<Vec<_>>().as_slice()) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {}", message);
            ExitCode::from(2)
        }
    }
}

#[derive(Default)]
struct Options {
    quiet: bool,
    optimize: Option<neura_ir::eir::OptimizeMode>,
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
        println!("npack {}", VERSION);
        return Ok(ExitCode::SUCCESS);
    }

    let mut directory = PathBuf::from(".");
    let mut options = Options::default();
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        if let Some(value) = arg.strip_prefix("--opt=") {
            options.optimize = Some(
                neura_ir::eir::OptimizeMode::parse(value)
                    .ok_or_else(|| format!("unknown optimisation mode `{}`", value))?,
            );
        } else if arg == "--quiet" {
            options.quiet = true;
        } else if arg.starts_with('-') {
            return Err(format!("unknown option `{}`", arg));
        } else if directory == Path::new(".") {
            directory = PathBuf::from(arg);
        } else {
            return Err(format!("unexpected argument `{}`", arg));
        }
    }

    match command.as_str() {
        "init" => cmd_init(&directory, &options),
        "install" => cmd_install(&directory, &options),
        "build" => cmd_build(&directory, &options).map(|_| ExitCode::SUCCESS),
        "run" => cmd_run(&directory, &options),
        "test" => cmd_test(&directory, &options),
        other => Err(format!("unknown command `{}`\n\n{}", other, USAGE)),
    }
}

// ---------------------------------------------------------------------------
// Semver
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    fn parse(text: &str) -> Result<Version, String> {
        let text = text.trim();
        let core = text.split(['-', '+']).next().unwrap_or(text);
        let mut parts = core.split('.');
        let mut next = |what: &str| -> Result<u64, String> {
            match parts.next() {
                Some(p) if !p.is_empty() => p
                    .parse::<u64>()
                    .map_err(|_| format!("invalid {} in version `{}`", what, text)),
                _ => Err(format!("invalid version `{}` (expected major.minor.patch)", text)),
            }
        };
        let major = next("major")?;
        let minor = next("minor")?;
        let patch = next("patch")?;
        if parts.next().is_some() {
            return Err(format!("invalid version `{}` (too many components)", text));
        }
        Ok(Version { major, minor, patch })
    }

    fn text(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Requirement {
    Any,
    Exact(Version),
    Caret(Version),
    Tilde(Version),
}

fn parse_requirement(text: &str) -> Result<Requirement, String> {
    let text = text.trim();
    if text.is_empty() || text == "*" {
        return Ok(Requirement::Any);
    }
    if let Some(rest) = text.strip_prefix('^') {
        return Ok(Requirement::Caret(Version::parse(rest)?));
    }
    if let Some(rest) = text.strip_prefix('~') {
        return Ok(Requirement::Tilde(Version::parse(rest)?));
    }
    Ok(Requirement::Exact(Version::parse(text)?))
}

fn satisfies(requirement: &Requirement, version: &Version) -> bool {
    match requirement {
        Requirement::Any => true,
        Requirement::Exact(v) => v == version,
        Requirement::Caret(base) => {
            if version < base {
                return false;
            }
            if base.major > 0 {
                version.major == base.major
            } else if base.minor > 0 {
                version.major == 0 && version.minor == base.minor
            } else {
                *version == *base
            }
        }
        Requirement::Tilde(base) => {
            version >= base && version.major == base.major && version.minor == base.minor
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest and lockfile
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct Dependency {
    name: String,
    requirement: Requirement,
    path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Manifest {
    name: String,
    version: Version,
    entry: String,
    dependencies: Vec<Dependency>,
}

impl Manifest {
    fn load(directory: &Path) -> Result<Manifest, String> {
        let path = directory.join(MANIFEST);
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
        Manifest::parse(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

    fn parse(text: &str) -> Result<Manifest, String> {
        let mut name = None;
        let mut version = None;
        let mut entry = None;
        let mut dependencies = Vec::new();
        let mut section = String::new();

        for (index, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                let header = rest
                    .strip_suffix(']')
                    .ok_or_else(|| format!("line {}: unterminated section header", index + 1))?;
                section = header.trim().to_string();
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: expected `key = value`", index + 1))?;
            let key = key.trim();
            let value = value.trim();
            match section.as_str() {
                "package" => match key {
                    "name" => name = Some(quoted(value, index)?),
                    "version" => version = Some(Version::parse(&quoted(value, index)?)?),
                    "entry" => entry = Some(quoted(value, index)?),
                    _ => return Err(format!("line {}: unknown package key `{}`", index + 1, key)),
                },
                "dependencies" => {
                    dependencies.push(parse_dependency(key, value, index)?);
                }
                other => {
                    return Err(format!(
                        "line {}: unsupported section `{}`",
                        index + 1,
                        other
                    ))
                }
            }
        }

        let name = name.ok_or("manifest is missing `package.name`")?;
        let version = version.ok_or("manifest is missing `package.version`")?;
        if name.is_empty() || !name.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false) {
            return Err(format!("invalid package name `{}`", name));
        }
        dependencies.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Manifest {
            name,
            version,
            entry: entry.unwrap_or_else(|| "src/main.neura".to_string()),
            dependencies,
        })
    }

    fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("[package]\n");
        out.push_str(&format!("name = \"{}\"\n", self.name));
        out.push_str(&format!("version = \"{}\"\n", self.version.text()));
        out.push_str(&format!("entry = \"{}\"\n", self.entry));
        if !self.dependencies.is_empty() {
            out.push_str("\n[dependencies]\n");
            for dep in &self.dependencies {
                match &dep.path {
                    Some(path) => out.push_str(&format!("{} = {{ path = \"{}\" }}\n", dep.name, path)),
                    None => out.push_str(&format!("{} = \"{}\"\n", dep.name, render_requirement(&dep.requirement))),
                }
            }
        }
        out
    }
}

fn strip_comment(line: &str) -> &str {
    // Manifest strings never contain `#` in this format, so a simple scan is
    // unambiguous.
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn quoted(value: &str, index: usize) -> Result<String, String> {
    let trimmed = value.trim();
    trimmed
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .map(|v| v.to_string())
        .ok_or_else(|| format!("line {}: expected a quoted string", index + 1))
}

fn render_requirement(requirement: &Requirement) -> String {
    match requirement {
        Requirement::Any => "*".to_string(),
        Requirement::Exact(v) => v.text(),
        Requirement::Caret(v) => format!("^{}", v.text()),
        Requirement::Tilde(v) => format!("~{}", v.text()),
    }
}

fn parse_dependency(name: &str, value: &str, index: usize) -> Result<Dependency, String> {
    if name.is_empty() {
        return Err(format!("line {}: empty dependency name", index + 1));
    }
    if value.starts_with('{') {
        let inner = value
            .strip_prefix('{')
            .and_then(|v| v.strip_suffix('}'))
            .ok_or_else(|| format!("line {}: malformed inline table", index + 1))?;
        let mut requirement = Requirement::Any;
        let mut path = None;
        for field in inner.split(',') {
            let field = field.trim();
            if field.is_empty() {
                continue;
            }
            let (key, raw) = field
                .split_once('=')
                .ok_or_else(|| format!("line {}: malformed inline table field `{}`", index + 1, field))?;
            let raw = quoted(raw.trim(), index)?;
            match key.trim() {
                "path" => path = Some(raw),
                "version" => requirement = parse_requirement(&raw)?,
                other => return Err(format!("line {}: unknown dependency key `{}`", index + 1, other)),
            }
        }
        return Ok(Dependency { name: name.to_string(), requirement, path });
    }
    Ok(Dependency {
        name: name.to_string(),
        requirement: parse_requirement(&quoted(value, index)?)?,
        path: None,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Locked {
    name: String,
    version: Version,
    source: String,
}

fn render_lock(packages: &[Locked]) -> String {
    let mut sorted = packages.to_vec();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut out = String::from("# generated by npack; do not edit\nversion = 1\n");
    for p in sorted {
        out.push_str("\n[[package]]\n");
        out.push_str(&format!("name = \"{}\"\n", p.name));
        out.push_str(&format!("version = \"{}\"\n", p.version.text()));
        out.push_str(&format!("source = \"{}\"\n", p.source));
    }
    out
}

fn cache_root() -> PathBuf {
    if let Some(home) = std::env::var_os("NPACK_HOME") {
        return PathBuf::from(home);
    }
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".npack"),
        None => PathBuf::from(".npack"),
    }
}

/// Resolve every dependency, recursively following local path dependencies.
fn resolve(manifest: &Manifest, root: &Path, cache: &Path) -> Result<Vec<Locked>, String> {
    let mut locked: BTreeMap<String, Locked> = BTreeMap::new();
    let mut visiting: Vec<String> = Vec::new();
    resolve_into(manifest, root, cache, &mut locked, &mut visiting)?;
    Ok(locked.into_values().collect())
}

fn resolve_into(
    manifest: &Manifest,
    root: &Path,
    cache: &Path,
    locked: &mut BTreeMap<String, Locked>,
    visiting: &mut Vec<String>,
) -> Result<(), String> {
    if visiting.contains(&manifest.name) {
        return Err(format!(
            "dependency cycle: {}",
            visiting.iter().chain(std::iter::once(&manifest.name)).cloned().collect::<Vec<_>>().join(" -> ")
        ));
    }
    visiting.push(manifest.name.clone());
    for dep in &manifest.dependencies {
        match &dep.path {
            Some(path) => {
                let location = root.join(path);
                let dep_manifest = Manifest::load(&location)
                    .map_err(|e| format!("dependency `{}`: {}", dep.name, e))?;
                if dep_manifest.name != dep.name {
                    return Err(format!(
                        "dependency `{}` points at a package named `{}`",
                        dep.name, dep_manifest.name
                    ));
                }
                if !satisfies(&dep.requirement, &dep_manifest.version) {
                    return Err(format!(
                        "dependency `{}` requires {} but {} provides {}",
                        dep.name,
                        render_requirement(&dep.requirement),
                        location.display(),
                        dep_manifest.version.text()
                    ));
                }
                let entry = Locked {
                    name: dep_manifest.name.clone(),
                    version: dep_manifest.version,
                    source: format!("path:{}", path),
                };
                if let Some(existing) = locked.get(&dep.name) {
                    if existing.version != entry.version || existing.source != entry.source {
                        return Err(format!(
                            "dependency `{}` is required at two different versions",
                            dep.name
                        ));
                    }
                } else {
                    locked.insert(dep.name.clone(), entry);
                    resolve_into(&dep_manifest, &location, cache, locked, visiting)?;
                }
            }
            None => {
                if locked.contains_key(&dep.name) {
                    continue;
                }
                let version = highest_cached(cache, &dep.name, &dep.requirement)?;
                locked.insert(
                    dep.name.clone(),
                    Locked {
                        name: dep.name.clone(),
                        version,
                        source: "cache".to_string(),
                    },
                );
            }
        }
    }
    visiting.pop();
    Ok(())
}

fn highest_cached(cache: &Path, name: &str, requirement: &Requirement) -> Result<Version, String> {
    let dir = cache.join(name);
    let mut best = None;
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let text = entry.file_name().to_string_lossy().to_string();
            if let Ok(version) = Version::parse(&text) {
                if satisfies(requirement, &version) && best.map(|b| version > b).unwrap_or(true) {
                    best = Some(version);
                }
            }
        }
    }
    best.ok_or_else(|| {
        format!(
            "no cached package `{}` matching {} (looked in {}); network access is not available in this build",
            name,
            render_requirement(requirement),
            dir.display()
        )
    })
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

struct Built {
    artifact: PathBuf,
    program: neura_ir::eir::EirProgram,
}

fn build(directory: &Path, options: &Options) -> Result<Built, String> {
    let manifest = Manifest::load(directory)?;
    verify_lockfile(&manifest, directory)?;

    let entry = directory.join(&manifest.entry);
    let source = fs::read_to_string(&entry)
        .map_err(|e| format!("cannot read {}: {}", entry.display(), e))?;
    let mut policies = CompilePolicies::default();
    if let Some(mode) = options.optimize {
        policies.optimize = mode;
    }
    let out = neura_compiler::compile(&source, policies);
    if neura_compiler::has_errors(&out.diags) {
        let rendered = render_all(&out.diags, &source);
        eprint!("{}", rendered);
        if !rendered.ends_with('\n') {
            eprintln!();
        }
        return Err(format!("{} failed to compile", entry.display()));
    }

    let mut program = out.eir.clone();
    let report = neura_optimizer::optimize(&mut program);
    let text = artifact_text(&CompileOutput { eir: program.clone(), ..out }, report.instructions_after);

    let target = directory.join("target");
    fs::create_dir_all(&target)
        .map_err(|e| format!("cannot create {}: {}", target.display(), e))?;
    let artifact = target.join(format!("{}.neurabc", manifest.name));
    fs::write(&artifact, &text)
        .map_err(|e| format!("cannot write {}: {}", artifact.display(), e))?;

    if !options.quiet {
        println!(
            "built {} ({} -> {} instructions)",
            artifact.display(),
            report.instructions_before,
            report.instructions_after
        );
    }
    Ok(Built { artifact, program })
}

/// Refuse to build from a manifest whose dependencies are not what the lock
/// records: builds must be reproducible.
fn verify_lockfile(manifest: &Manifest, directory: &Path) -> Result<(), String> {
    if manifest.dependencies.is_empty() {
        return Ok(());
    }
    let expected = render_lock(&resolve(manifest, directory, &cache_root())?);
    let path = directory.join(LOCKFILE);
    let actual = fs::read_to_string(&path)
        .map_err(|_| format!("missing {}; run `npack install`", LOCKFILE))?;
    if actual != expected {
        return Err(format!("{} is out of date; run `npack install`", LOCKFILE));
    }
    Ok(())
}

fn artifact_text(output: &CompileOutput, instructions: usize) -> String {
    format!(
        "neurabc 1\nsource-hash {:016x}\nprogram {}\noptimize {}\ndevice {}\nprecision {}\ninstructions {}\n---\n{}",
        output.eir.source_hash,
        output.eir.name,
        output.eir.policies.optimize.name(),
        output.eir.policies.device.name(),
        output.eir.policies.precision.name(),
        instructions,
        neura_formatter::format_program(&output.program),
    )
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_init(directory: &Path, options: &Options) -> Result<ExitCode, String> {
    let manifest_path = directory.join(MANIFEST);
    if manifest_path.exists() {
        return Err(format!("{} already exists", manifest_path.display()));
    }
    fs::create_dir_all(directory)
        .map_err(|e| format!("cannot create {}: {}", directory.display(), e))?;
    let name = directory
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .filter(|n| !n.is_empty() && n.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false))
        .map(|n| n.replace('-', "_"))
        .unwrap_or_else(|| "neura_package".to_string());

    let manifest = Manifest {
        name: name.clone(),
        version: Version { major: 0, minor: 1, patch: 0 },
        entry: "src/main.neura".to_string(),
        dependencies: Vec::new(),
    };
    fs::write(&manifest_path, manifest.render())
        .map_err(|e| format!("cannot write {}: {}", manifest_path.display(), e))?;

    let src = directory.join("src");
    fs::create_dir_all(&src).map_err(|e| format!("cannot create {}: {}", src.display(), e))?;
    let main = src.join("main.neura");
    fs::write(&main, format!("fn main():\n    print(\"hello from {}\")\n", name))
        .map_err(|e| format!("cannot write {}: {}", main.display(), e))?;
    let tests = directory.join("tests");
    fs::create_dir_all(&tests).map_err(|e| format!("cannot create {}: {}", tests.display(), e))?;

    if !options.quiet {
        println!("created {} and {}", manifest_path.display(), main.display());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_install(directory: &Path, options: &Options) -> Result<ExitCode, String> {
    let manifest = Manifest::load(directory)?;
    let locked = resolve(&manifest, directory, &cache_root())?;
    let text = render_lock(&locked);
    let path = directory.join(LOCKFILE);
    fs::write(&path, &text).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
    if !options.quiet {
        println!("wrote {} ({} package(s))", path.display(), locked.len());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_build(directory: &Path, options: &Options) -> Result<Built, String> {
    build(directory, options)
}

fn cmd_run(directory: &Path, options: &Options) -> Result<ExitCode, String> {
    let built = build(directory, options)?;
    let vm = VmOptions { trace: false, quiet: false, training: false, profile: false };
    match run_program(&built.program, vm) {
        Ok(_) => Ok(ExitCode::SUCCESS),
        Err(message) => {
            eprintln!("runtime error: {}", message);
            Ok(ExitCode::from(1))
        }
    }
}

fn cmd_test(directory: &Path, options: &Options) -> Result<ExitCode, String> {
    let built = build(directory, options)?;
    let tests_dir = directory.join("tests");
    let mut files: Vec<PathBuf> = match fs::read_dir(&tests_dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("neura"))
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "no tests found in {} (build produced {})",
            tests_dir.display(),
            built.artifact.display()
        ));
    }

    let mut failures = 0usize;
    for file in &files {
        let source = fs::read_to_string(file)
            .map_err(|e| format!("cannot read {}: {}", file.display(), e))?;
        let out = neura_compiler::compile(&source, CompilePolicies::default());
        if neura_compiler::has_errors(&out.diags) {
            eprint!("{}", render_all(&out.diags, &source));
            println!("FAIL {}", file.display());
            failures += 1;
            continue;
        }
        let mut program = out.eir;
        neura_optimizer::optimize(&mut program);
        let vm = VmOptions { trace: false, quiet: true, training: false, profile: false };
        match run_program(&program, vm) {
            Ok(_) => println!("PASS {}", file.display()),
            Err(message) => {
                println!("FAIL {}: {}", file.display(), message);
                failures += 1;
            }
        }
    }
    println!(
        "{} passed, {} failed",
        files.len().saturating_sub(failures),
        failures
    );
    Ok(if failures == 0 { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(tag: &str) -> PathBuf {
        let unique = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("npack-{}-{}-{}", std::process::id(), tag, unique));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn manifest_round_trips_through_the_parser() {
        let text = "\
[package]
name = \"demo\"
version = \"1.2.3\"
entry = \"src/app.neura\"

[dependencies]
alpha = { path = \"../alpha\" }
beta = \"^0.2.0\"
";
        let manifest = Manifest::parse(text).unwrap();
        assert_eq!(manifest.name, "demo");
        assert_eq!(manifest.version, Version { major: 1, minor: 2, patch: 3 });
        assert_eq!(manifest.entry, "src/app.neura");
        assert_eq!(manifest.dependencies.len(), 2);
        assert_eq!(manifest.dependencies[0].name, "alpha");
        assert_eq!(manifest.dependencies[0].path.as_deref(), Some("../alpha"));
        assert_eq!(manifest.dependencies[1].requirement, Requirement::Caret(Version { major: 0, minor: 2, patch: 0 }));

        let reparsed = Manifest::parse(&manifest.render()).unwrap();
        assert_eq!(manifest, reparsed);
    }

    #[test]
    fn semver_requirements_match_expected_versions() {
        let exact = parse_requirement("1.2.3").unwrap();
        assert!(satisfies(&exact, &Version { major: 1, minor: 2, patch: 3 }));
        assert!(!satisfies(&exact, &Version { major: 1, minor: 2, patch: 4 }));

        let caret = parse_requirement("^1.2.3").unwrap();
        assert!(satisfies(&caret, &Version { major: 1, minor: 9, patch: 0 }));
        assert!(!satisfies(&caret, &Version { major: 2, minor: 0, patch: 0 }));

        let zero = parse_requirement("^0.2.3").unwrap();
        assert!(satisfies(&zero, &Version { major: 0, minor: 2, patch: 9 }));
        assert!(!satisfies(&zero, &Version { major: 0, minor: 3, patch: 0 }));

        let tilde = parse_requirement("~1.2.3").unwrap();
        assert!(satisfies(&tilde, &Version { major: 1, minor: 2, patch: 9 }));
        assert!(!satisfies(&tilde, &Version { major: 1, minor: 3, patch: 0 }));

        assert!(satisfies(&Requirement::Any, &Version { major: 9, minor: 9, patch: 9 }));
        assert!(Version::parse("1.2").is_err());
    }

    #[test]
    fn init_build_and_run_produce_a_real_artifact() {
        let dir = temp_dir("build");
        cmd_init(&dir, &Options { quiet: true, optimize: None }).unwrap();
        assert!(dir.join(MANIFEST).exists());
        assert!(dir.join("src/main.neura").exists());

        let built = build(&dir, &Options { quiet: true, optimize: None }).unwrap();
        assert!(built.artifact.exists());
        let text = fs::read_to_string(&built.artifact).unwrap();
        assert!(text.starts_with("neurabc 1\n"));
        assert!(text.contains("source-hash "));
        let instructions: usize = text
            .lines()
            .find_map(|line| line.strip_prefix("instructions "))
            .and_then(|n| n.trim().parse().ok())
            .expect("artifact has no instruction count");
        assert!(instructions > 0);
        assert!(text.contains("print(\"hello from "));

        let code = cmd_run(&dir, &Options { quiet: true, optimize: None }).unwrap();
        assert_eq!(format!("{:?}", code), format!("{:?}", ExitCode::SUCCESS));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn registry_dependency_without_cache_is_reported_honestly() {
        let dir = temp_dir("registry");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join(MANIFEST),
            "[package]\nname = \"needs_registry\"\nversion = \"0.1.0\"\n\n[dependencies]\nremote = \"^1.0.0\"\n",
        )
        .unwrap();
        fs::write(dir.join("src/main.neura"), "fn main():\n    print(1)\n").unwrap();

        let cache = dir.join("empty-cache");
        let manifest = Manifest::load(&dir).unwrap();
        let error = resolve(&manifest, &dir, &cache).unwrap_err();
        assert!(error.contains("network access is not available"), "{}", error);
        assert!(error.contains("remote"), "{}", error);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_lockfile_blocks_the_build() {
        let dir = temp_dir("lock");
        let dep = dir.join("dep");
        fs::create_dir_all(dep.join("src")).unwrap();
        fs::write(
            dep.join(MANIFEST),
            "[package]\nname = \"dep\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::write(dep.join("src/main.neura"), "fn main():\n    print(1)\n").unwrap();

        let pkg = dir.join("pkg");
        fs::create_dir_all(pkg.join("src")).unwrap();
        fs::write(
            pkg.join(MANIFEST),
            "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\n\n[dependencies]\ndep = { path = \"../dep\" }\n",
        )
        .unwrap();
        fs::write(pkg.join("src/main.neura"), "fn main():\n    print(2)\n").unwrap();

        let error = match build(&pkg, &Options { quiet: true, optimize: None }) {
            Ok(_) => panic!("expected the build to reject a missing lockfile"),
            Err(error) => error,
        };
        assert!(error.contains("npack install"), "{}", error);

        cmd_install(&pkg, &Options { quiet: true, optimize: None }).unwrap();
        let built = build(&pkg, &Options { quiet: true, optimize: None }).unwrap();
        assert!(built.artifact.exists());
        let lock = fs::read_to_string(pkg.join(LOCKFILE)).unwrap();
        assert!(lock.contains("name = \"dep\""));
        assert!(lock.contains("source = \"path:../dep\""));
        let _ = fs::remove_dir_all(&dir);
    }
}
