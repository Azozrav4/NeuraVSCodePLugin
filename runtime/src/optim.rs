//! Optimizers operating on graph leaves (`TRef` parameters).

use crate::value::TRef;
use neura_backends::tensor::Tensor;
use neura_ir::types::DType;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub enum OptKind {
    Sgd { momentum: f64 },
    Adam { beta1: f64, beta2: f64, eps: f64 },
    AdamW { beta1: f64, beta2: f64, eps: f64, weight_decay: f64 },
    RmsProp { alpha: f64, eps: f64, momentum: f64 },
}

#[derive(Default)]
struct Slot {
    m: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
    t: u64,
}

pub struct Optimizer {
    pub kind: OptKind,
    pub lr: f64,
    pub params: Vec<TRef>,
    slots: Vec<Slot>,
}

impl Optimizer {
    pub fn new(kind: OptKind, lr: f64, params: Vec<TRef>) -> Optimizer {
        let n = params.len();
        Optimizer { kind, lr, params, slots: (0..n).map(|_| Slot::default()).collect() }
    }

    pub fn zero_grad(&mut self) {
        for p in &self.params {
            p.borrow_mut().grad = None;
        }
    }

    pub fn step(&mut self) -> Result<(), String> {
        if self.params.len() != self.slots.len() {
            return Err("optimizer parameter/state size mismatch".into());
        }
        for (i, p) in self.params.iter().enumerate() {
            let (grad, mut data) = {
                let node = p.borrow();
                match &node.grad {
                    Some(g) => (g.clone(), node.data.clone()),
                    None => continue,
                }
            };
            if grad.numel() != data.numel() {
                return Err(format!(
                    "parameter shape {:?} does not match gradient shape {:?}",
                    data.shape, grad.shape
                ));
            }
            let n = data.numel();
            let mut vals = data.to_f64_vec();
            let g: Vec<f64> = (0..n).map(|k| grad.f64_at(k)).collect();
            let slot = &mut self.slots[i];
            match self.kind {
                OptKind::Sgd { momentum } => {
                    if momentum > 0.0 {
                        if slot.m.is_none() {
                            slot.m = Some(vec![0.0; n]);
                        }
                        let m = slot.m.as_mut().unwrap();
                        for k in 0..n {
                            m[k] = momentum * m[k] + g[k];
                            vals[k] -= self.lr * m[k];
                        }
                    } else {
                        for k in 0..n {
                            vals[k] -= self.lr * g[k];
                        }
                    }
                }
                OptKind::Adam { beta1, beta2, eps } => {
                    if slot.m.is_none() {
                        slot.m = Some(vec![0.0; n]);
                        slot.v = Some(vec![0.0; n]);
                    }
                    slot.t += 1;
                    let m = slot.m.as_mut().unwrap();
                    let v = slot.v.as_mut().unwrap();
                    let bc1 = 1.0 - beta1.powi(slot.t as i32);
                    let bc2 = 1.0 - beta2.powi(slot.t as i32);
                    for k in 0..n {
                        m[k] = beta1 * m[k] + (1.0 - beta1) * g[k];
                        v[k] = beta2 * v[k] + (1.0 - beta2) * g[k] * g[k];
                        let mh = m[k] / bc1;
                        let vh = v[k] / bc2;
                        vals[k] -= self.lr * mh / (vh.sqrt() + eps);
                    }
                }
                OptKind::AdamW { beta1, beta2, eps, weight_decay } => {
                    if slot.m.is_none() {
                        slot.m = Some(vec![0.0; n]);
                        slot.v = Some(vec![0.0; n]);
                    }
                    slot.t += 1;
                    let m = slot.m.as_mut().unwrap();
                    let v = slot.v.as_mut().unwrap();
                    let bc1 = 1.0 - beta1.powi(slot.t as i32);
                    let bc2 = 1.0 - beta2.powi(slot.t as i32);
                    for k in 0..n {
                        vals[k] -= self.lr * weight_decay * vals[k];
                        m[k] = beta1 * m[k] + (1.0 - beta1) * g[k];
                        v[k] = beta2 * v[k] + (1.0 - beta2) * g[k] * g[k];
                        let mh = m[k] / bc1;
                        let vh = v[k] / bc2;
                        vals[k] -= self.lr * mh / (vh.sqrt() + eps);
                    }
                }
                OptKind::RmsProp { alpha, eps, momentum } => {
                    if slot.v.is_none() {
                        slot.v = Some(vec![0.0; n]);
                        if momentum > 0.0 {
                            slot.m = Some(vec![0.0; n]);
                        }
                    }
                    let v = slot.v.as_mut().unwrap();
                    let m = slot.m.clone();
                    let mut m_new = m.clone();
                    for k in 0..n {
                        v[k] = alpha * v[k] + (1.0 - alpha) * g[k] * g[k];
                        let upd = self.lr * g[k] / (v[k].sqrt() + eps);
                        if let Some(mm) = m_new.as_mut() {
                            mm[k] = momentum * mm[k] + upd;
                            vals[k] -= mm[k];
                        } else {
                            vals[k] -= upd;
                        }
                    }
                    if m_new.is_some() {
                        slot.m = m_new;
                    }
                }
            }
            let dtype = data.dtype;
            data = Tensor::from_f64(dtype, data.shape.clone(), vals)?;
            if dtype.is_int() {
                data = data.cast(DType::F64).cast(dtype);
            }
            p.borrow_mut().data = data;
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        let kind = match self.kind {
            OptKind::Sgd { momentum } => format!("SGD(lr={}, momentum={})", self.lr, momentum),
            OptKind::Adam { beta1, beta2, eps } => {
                format!("Adam(lr={}, betas=({}, {}), eps={})", self.lr, beta1, beta2, eps)
            }
            OptKind::AdamW { beta1, beta2, eps, weight_decay } => format!(
                "AdamW(lr={}, betas=({}, {}), eps={}, weight_decay={})",
                self.lr, beta1, beta2, eps, weight_decay
            ),
            OptKind::RmsProp { alpha, eps, momentum } => format!(
                "RMSProp(lr={}, alpha={}, eps={}, momentum={})",
                self.lr, alpha, eps, momentum
            ),
        };
        format!("{} params={}", kind, self.params.len())
    }
}

pub fn shared(kind: OptKind, lr: f64, params: Vec<TRef>) -> Rc<RefCell<Optimizer>> {
    Rc::new(RefCell::new(Optimizer::new(kind, lr, params)))
}
