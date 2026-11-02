//! The Neura compiler front end and lowering pipeline.
//!
//! lexer -> parser -> type checking -> ownership analysis -> EIR lowering.
//! Optimisation passes (NAOE) and backend selection consume the EIR program.

pub mod ast;
pub mod builtins;
pub mod diagnostics;
pub mod lexer;
pub mod lowering;
pub mod ownership;
pub mod parser;
pub mod typecheck;

pub use diagnostics::{Diagnostic, Severity, Span};
pub use neura_ir::eir::{CompilePolicies, EirProgram};

pub struct CompileResult {
    pub program: ast::Program,
    pub module: neura_ir::nir::NirModule,
    pub diags: Vec<Diagnostic>,
    pub report: typecheck::TypeReport,
}

pub struct CompileOutput {
    pub program: ast::Program,
    pub nir: neura_ir::nir::NirModule,
    pub eir: EirProgram,
    pub diags: Vec<Diagnostic>,
    pub report: typecheck::TypeReport,
}

/// Lex, parse and type-check a Neura source file (front end only).
pub fn compile_source(source: &str) -> CompileResult {
    let (program, diags) = parser::parse(source);
    let checked = typecheck::check(&program);
    let mut all = diags;
    all.extend(checked.diags);
    CompileResult {
        program,
        module: checked.module,
        diags: all,
        report: checked.report,
    }
}

/// Full front end: parse, type-check, ownership-check and lower to EIR.
pub fn compile(source: &str, policies: CompilePolicies) -> CompileOutput {
    let (program, parse_diags) = parser::parse(source);
    let checked = typecheck::check(&program);
    let own = ownership::check(&checked.module);
    let lowered = lowering::lower(&checked.module, policies, lowering::hash_source(source));
    let mut diags = parse_diags;
    diags.extend(checked.diags);
    diags.extend(own);
    diags.extend(lowered.diags);
    CompileOutput {
        program,
        eir: lowered.program,
        nir: checked.module,
        diags,
        report: checked.report,
    }
}

/// True when any diagnostic is an error.
pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.severity == Severity::Error)
}
