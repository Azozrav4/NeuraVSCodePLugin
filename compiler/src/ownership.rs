//! Ownership, borrowing and lifetime analysis (M2).
//!
//! Neura tensors, lists, structs and library objects are *owned* values.
//! Scalars (`Int`, `Float`, `Bool`, `Unit`) are copyable. The rules enforced
//! here are:
//!
//! * a value can be moved once; using it afterwards is a use-after-move error;
//! * `&x` creates a short-lived shared borrow, `&mut x` an exclusive borrow;
//! * a mutable borrow requires the binding to be declared `mut`;
//! * a value cannot be moved while it is borrowed;
//! * passing a tensor to a user-defined function transfers ownership, while
//!   standard-library calls only borrow their arguments (so `model.forward(x)`
//!   and `optimizer.step()` do not consume the model);
//! * returning a local tensor transfers it out of the function.
//!
//! Borrows are transient values (the language has no reference types), so a
//! borrow is live only for the duration of the enclosing call expression.

use crate::diagnostics::{Diagnostic, Span};
use neura_ir::nir::{LocalId, NirBlock, NirExpr, NirFunction, NirModule, NirStmt};
use neura_ir::types::Ty;
use std::collections::HashMap;

#[derive(Clone, Copy, Default, Debug)]
struct LocalState {
    moved: bool,
}

struct FnChecker<'a> {
    f: &'a NirFunction,
    state: HashMap<LocalId, LocalState>,
    diags: Vec<Diagnostic>,
    /// Locals borrowed by the call expression currently being checked.
    borrows: Vec<LocalId>,
}

pub fn check(module: &NirModule) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    for f in &module.functions {
        let mut c = FnChecker {
            f,
            state: HashMap::new(),
            diags: Vec::new(),
            borrows: Vec::new(),
        };
        c.walk_block(&f.body);
        diags.extend(c.diags);
    }
    diags
}

fn is_copy(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::Bool | Ty::Unit | Ty::Never)
}

fn span_of(s: (usize, usize)) -> Span {
    Span { start: s.0, end: s.1, line: 0, col: 0 }
}

impl<'a> FnChecker<'a> {
    fn local_ty(&self, id: LocalId) -> Ty {
        self.f
            .locals
            .get(id as usize)
            .map(|l| l.ty.clone())
            .unwrap_or(Ty::Any)
    }

    fn local_name(&self, id: LocalId) -> String {
        self.f
            .locals
            .get(id as usize)
            .map(|l| l.name.clone())
            .unwrap_or_else(|| format!("#{}", id))
    }

    fn is_mut(&self, id: LocalId) -> bool {
        self.f.locals.get(id as usize).map(|l| l.mutable).unwrap_or(false)
    }

    fn read(&mut self, id: LocalId, span: (usize, usize)) {
        if self.state.get(&id).map(|s| s.moved).unwrap_or(false) {
            self.diags.push(
                Diagnostic::error(
                    "E0401",
                    format!("use of moved value `{}`", self.local_name(id)),
                    Some(span_of(span)),
                )
                .help("clone or re-bind the tensor before using it again"),
            );
        }
    }

    fn move_out(&mut self, id: LocalId, span: (usize, usize)) {
        if self.borrows.contains(&id) {
            self.diags.push(Diagnostic::error(
                "E0402",
                format!(
                    "cannot move `{}` while it is borrowed by the same call",
                    self.local_name(id)
                ),
                Some(span_of(span)),
            ));
        }
        self.read(id, span);
        self.state.entry(id).or_default().moved = true;
    }

    fn walk_block(&mut self, b: &NirBlock) {
        for s in &b.stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, s: &NirStmt) {
        match s {
            NirStmt::Let { local, expr, span } => {
                self.walk_expr(expr);
                // `let y = x` transfers ownership of non-Copy values.
                if let NirExpr::Local(src) = expr {
                    if !is_copy(&self.local_ty(*src)) && src != local {
                        self.move_out(*src, *span);
                    }
                }
                self.state.insert(*local, LocalState::default());
            }
            NirStmt::Assign { local, expr, span } => {
                self.walk_expr(expr);
                if let NirExpr::Local(src) = expr {
                    if !is_copy(&self.local_ty(*src)) && src != local {
                        self.move_out(*src, *span);
                    }
                }
                self.state.insert(*local, LocalState::default());
            }
            NirStmt::Expr(e) => self.walk_expr(e),
            NirStmt::If { cond, then_block, else_block, .. } => {
                self.walk_expr(cond);
                let before = self.state.clone();
                self.walk_block(then_block);
                let then_state = self.state.clone();
                self.state = before.clone();
                if let Some(b) = else_block {
                    self.walk_block(b);
                }
                // After the branch a local is moved if it was moved on either path.
                let mut merged = HashMap::new();
                for (id, st) in before {
                    let t = then_state.get(&id).copied().unwrap_or_default();
                    let e = self.state.get(&id).copied().unwrap_or_default();
                    merged.insert(
                        id,
                        LocalState { moved: t.moved || e.moved || st.moved },
                    );
                }
                self.state = merged;
            }
            NirStmt::While { cond, body, .. } => {
                self.walk_expr(cond);
                let before = self.state.clone();
                self.walk_block(body);
                // The loop may run zero times, so keep everything that was
                // moved before, plus anything moved on every iteration.
                for (id, st) in before {
                    if st.moved {
                        self.state.insert(id, st);
                    }
                }
            }
            NirStmt::For { iter, body, .. } => {
                self.walk_expr(iter);
                let before = self.state.clone();
                self.walk_block(body);
                for (id, st) in before {
                    if st.moved {
                        self.state.insert(id, st);
                    }
                }
            }
            NirStmt::Return(value, span) => {
                if let Some(e) = value {
                    self.walk_expr(e);
                    if let NirExpr::Local(id) = e {
                        if !is_copy(&self.local_ty(*id)) {
                            self.move_out(*id, *span);
                        }
                    }
                }
            }
            NirStmt::Break(_) | NirStmt::Continue(_) | NirStmt::Pass => {}
        }
    }

    fn walk_expr(&mut self, e: &NirExpr) {
        match e {
            NirExpr::Local(id) => self.read(*id, (0, 0)),
            NirExpr::Move(id) => self.move_out(*id, (0, 0)),
            NirExpr::Borrow { local, mutable } => {
                self.read(*local, (0, 0));
                if *mutable && !self.is_mut(*local) {
                    self.diags.push(
                        Diagnostic::error(
                            "E0406",
                            format!(
                                "cannot mutably borrow immutable variable `{}`",
                                self.local_name(*local)
                            ),
                            Some(span_of((0, 0))),
                        )
                        .help("declare it with `let mut`"),
                    );
                }
                self.borrows.push(*local);
            }
            NirExpr::Unary { expr, .. } => self.walk_expr(expr),
            NirExpr::Binary { lhs, rhs, .. } => {
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            NirExpr::List { elems, .. } => {
                for e in elems {
                    self.walk_expr(e);
                }
            }
            NirExpr::Call { callee, args, .. } => {
                let user_fn = matches!(**callee, NirExpr::FunctionRef { .. });
                self.walk_expr(callee);
                let saved = std::mem::take(&mut self.borrows);
                for a in args {
                    self.walk_expr(&a.expr);
                    let borrowed: Vec<LocalId> = self.borrows.clone();
                    self.borrows = saved.clone();
                    if user_fn {
                        // Passing an owned value to a user function moves it,
                        // unless the argument is explicitly borrowed.
                        if let NirExpr::Local(id) = &a.expr {
                            let ty = self.local_ty(*id);
                            if !is_copy(&ty) && !borrowed.contains(id) {
                                self.move_out(*id, (0, 0));
                            }
                        }
                    }
                }
                self.borrows = saved;
            }
            NirExpr::Attr { obj, .. } | NirExpr::Index { obj, .. } => {
                self.walk_expr(obj);
                if let NirExpr::Index { index, .. } = e {
                    self.walk_expr(index);
                }
            }
            NirExpr::TensorOp { args, dyn_dims, .. } => {
                for a in args {
                    self.walk_expr(a);
                }
                for d in dyn_dims {
                    self.walk_expr(d);
                }
            }
            NirExpr::Int(_)
            | NirExpr::Float(_)
            | NirExpr::Bool(_)
            | NirExpr::Str(_)
            | NirExpr::Unit
            | NirExpr::Module(_)
            | NirExpr::Builtin { .. }
            | NirExpr::FunctionRef { .. } => {}
        }
    }
}

/// Instruction index of the final read of a register, used by NAOE for buffer
/// reuse and by the runtime to release GPU buffers as soon as it is safe.
pub fn last_use_index(code: &[neura_ir::eir::Inst], reg: neura_ir::eir::Reg) -> Option<usize> {
    code.iter()
        .enumerate()
        .filter(|(_, inst)| inst.reads().contains(&reg))
        .map(|(i, _)| i)
        .next_back()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::typecheck::check as typecheck;

    fn ownership(src: &str) -> (Vec<Diagnostic>, NirModule) {
        let (p, d) = parse(src);
        assert!(d.is_empty(), "{:?}", d);
        let r = typecheck(&p);
        let d = check(&r.module);
        (d, r.module)
    }

    #[test]
    fn detects_use_after_move() {
        let (d, _) = ownership(
            "fn consume(x: Tensor[2]) -> Tensor[2]:\n    return x\n\nfn main():\n    a = Tensor([1.0, 2.0])\n    b = consume(a)\n    c = consume(a)\n",
        );
        assert!(d.iter().any(|x| x.code == "E0401"), "{:#?}", d);
    }

    #[test]
    fn accepts_borrowing_arguments() {
        let (d, _) = ownership(
            "fn main():\n    a = Tensor([1.0, 2.0])\n    b = a + a\n    print(b)\n",
        );
        assert!(d.is_empty(), "{:#?}", d);
    }

    #[test]
    fn rejects_mutable_borrow_of_immutable() {
        let (d, _) = ownership(
            "fn main():\n    let a = Tensor([1.0, 2.0])\n    b = &mut a\n    print(b)\n",
        );
        assert!(d.iter().any(|x| x.code == "E0406"), "{:#?}", d);
    }

    #[test]
    fn last_use_of_register() {
        use neura_ir::eir::Inst;
        let code = vec![
            Inst::ConstInt { dst: 0, v: 1 },
            Inst::Copy { dst: 1, src: 0 },
            Inst::Print(1),
        ];
        assert_eq!(last_use_index(&code, 0), Some(1));
        assert_eq!(last_use_index(&code, 1), Some(2));
    }
}
