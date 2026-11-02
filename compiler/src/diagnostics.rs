//! Structured compiler diagnostics with human and machine-readable output.

use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, col: usize) -> Span {
        Span { start, end, line, col }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            line: self.line,
            col: self.col,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    pub fn name(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(code: &str, message: impl Into<String>, span: Option<Span>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            span,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn warning(code: &str, message: impl Into<String>, span: Option<Span>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            span,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn note(mut self, n: impl Into<String>) -> Diagnostic {
        self.notes.push(n.into());
        self
    }

    pub fn help(mut self, h: impl Into<String>) -> Diagnostic {
        self.help = Some(h.into());
        self
    }

    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        let _ = write!(s, "\"severity\":\"{}\",", self.severity.name());
        let _ = write!(s, "\"code\":\"{}\",", escape(&self.code));
        let _ = write!(s, "\"message\":\"{}\"", escape(&self.message));
        if let Some(span) = self.span {
            let _ = write!(
                s,
                ",\"span\":{{\"line\":{},\"column\":{},\"start\":{},\"end\":{}}}",
                span.line, span.col, span.start, span.end
            );
        }
        if !self.notes.is_empty() {
            let notes: Vec<String> = self.notes.iter().map(|n| format!("\"{}\"", escape(n))).collect();
            let _ = write!(s, ",\"notes\":[{}]", notes.join(","));
        }
        if let Some(h) = &self.help {
            let _ = write!(s, ",\"help\":\"{}\"", escape(h));
        }
        s.push('}');
        s
    }

    pub fn render(&self, source: &str) -> String {
        let mut out = String::new();
        let _ = write!(out, "{}[{}]: {}", self.severity.name(), self.code, self.message);
        if let Some(span) = self.span {
            let _ = write!(out, "\n  --> {}:{}", span.line, span.col);
            if let Some(line) = source.lines().nth(span.line.saturating_sub(1)) {
                let _ = write!(out, "\n   |\n{:>3}| {}", span.line, line);
                let width = if span.end > span.start {
                    span.end.saturating_sub(span.start).max(1)
                } else {
                    1
                };
                let _ = write!(
                    out,
                    "\n   | {}{}",
                    " ".repeat(span.col.saturating_sub(1)),
                    "^".repeat(width.min(line.len().max(1)))
                );
            }
        }
        for n in &self.notes {
            let _ = write!(out, "\n  = note: {}", n);
        }
        if let Some(h) = &self.help {
            let _ = write!(out, "\n  = help: {}", h);
        }
        out
    }
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.severity == Severity::Error)
}

pub fn render_all(diags: &[Diagnostic], source: &str) -> String {
    diags
        .iter()
        .map(|d| d.render(source))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_with_snippet() {
        let d = Diagnostic::error("E0001", "unexpected token", Some(Span::new(4, 5, 1, 5)))
            .help("did you mean `let`?");
        let text = d.render("let x = @\n");
        assert!(text.contains("E0001"));
        assert!(text.contains("^"));
        assert!(text.contains("did you mean"));
    }

    #[test]
    fn json_is_structured() {
        let d = Diagnostic::error("E0001", "boom", Some(Span::new(0, 1, 1, 1)));
        let j = d.to_json();
        assert!(j.contains("\"severity\":\"error\""));
        assert!(j.contains("\"span\""));
    }
}
