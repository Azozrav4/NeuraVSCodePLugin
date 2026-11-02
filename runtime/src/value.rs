//! Runtime values and the tensor graph node used by autograd.
//!
//! Tensors are reference counted so that (a) the autograd graph can keep the
//! values it needs alive until their last dependent gradient is produced, and
//! (b) copies are cheap. Ownership *rules* are enforced statically by the
//! compiler; reference counting here is an implementation detail of the
//! runtime, not the language's memory-management strategy.

use crate::autograd::Vjp;
use crate::data::Dataset;
use crate::layers::Layer;
use crate::optim::Optimizer;
use neura_backends::tensor::Tensor;
use neura_ir::types::DType;
use std::cell::RefCell;
use std::rc::Rc;

pub type TRef = Rc<RefCell<TNode>>;

/// A tensor value in the execution graph.
pub struct TNode {
    pub data: Tensor,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub inputs: Vec<TRef>,
    pub vjp: Option<Vjp>,
}

impl TNode {
    pub fn leaf(data: Tensor, requires_grad: bool) -> TRef {
        Rc::new(RefCell::new(TNode {
            data,
            requires_grad,
            grad: None,
            inputs: Vec::new(),
            vjp: None,
        }))
    }

    pub fn derived(data: Tensor, inputs: Vec<TRef>, vjp: Vjp) -> TRef {
        let requires_grad = inputs.iter().any(|i| i.borrow().requires_grad);
        Rc::new(RefCell::new(TNode {
            data,
            requires_grad,
            grad: None,
            inputs,
            vjp: Some(vjp),
        }))
    }
}

pub fn tensor(data: Tensor) -> TRef {
    TNode::leaf(data, false)
}

pub fn trainable(data: Tensor) -> TRef {
    TNode::leaf(data, true)
}

/// A runtime value. Scalars are plain Rust values; aggregates are cheap to copy.
#[derive(Clone)]
pub enum Value {
    Unit,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Tensor(TRef),
    List(Rc<RefCell<Vec<Value>>>),
    Struct { name: String, fields: Vec<(String, Value)> },
    Layer(Rc<Layer>),
    Optimizer(Rc<RefCell<Optimizer>>),
    Dataset(Rc<Dataset>),
    Range(i64, i64),
}

impl Value {
    pub fn list(v: Vec<Value>) -> Value {
        Value::List(Rc::new(RefCell::new(v)))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "Unit",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::Tensor(_) => "Tensor",
            Value::List(_) => "List",
            Value::Struct { .. } => "struct",
            Value::Layer(_) => "Layer",
            Value::Optimizer(_) => "Optimizer",
            Value::Dataset(_) => "Dataset",
            Value::Range(..) => "range",
        }
    }

    pub fn as_f64(&self) -> Result<f64, String> {
        match self {
            Value::Float(v) => Ok(*v),
            Value::Int(v) => Ok(*v as f64),
            Value::Bool(v) => Ok(*v as i64 as f64),
            Value::Tensor(t) => t
                .borrow()
                .data
                .scalar_f64()
                .ok_or_else(|| "expected a scalar tensor".to_string()),
            other => Err(format!("expected a number, found {}", other.type_name())),
        }
    }

    pub fn as_i64(&self) -> Result<i64, String> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Float(v) => Ok(*v as i64),
            Value::Bool(v) => Ok(*v as i64),
            Value::Tensor(t) => {
                let b = t.borrow();
                if b.data.numel() != 1 {
                    return Err("expected a scalar tensor".into());
                }
                Ok(b.data.i64_at(0))
            }
            other => Err(format!("expected an integer, found {}", other.type_name())),
        }
    }

    pub fn as_bool(&self) -> Result<bool, String> {
        match self {
            Value::Bool(v) => Ok(*v),
            Value::Int(v) => Ok(*v != 0),
            Value::Float(v) => Ok(*v != 0.0),
            Value::Tensor(t) => {
                let b = t.borrow();
                if b.data.numel() != 1 {
                    return Err("expected a scalar tensor in condition".into());
                }
                Ok(b.data.f64_at(0) != 0.0)
            }
            other => Err(format!("expected a boolean, found {}", other.type_name())),
        }
    }

    pub fn as_tensor(&self) -> Result<TRef, String> {
        match self {
            Value::Tensor(t) => Ok(t.clone()),
            other => Err(format!("expected a tensor, found {}", other.type_name())),
        }
    }

    pub fn as_list(&self) -> Result<Rc<RefCell<Vec<Value>>>, String> {
        match self {
            Value::List(l) => Ok(l.clone()),
            other => Err(format!("expected a list, found {}", other.type_name())),
        }
    }

    pub fn as_str(&self) -> Result<String, String> {
        match self {
            Value::Str(s) => Ok(s.clone()),
            other => Err(format!("expected a string, found {}", other.type_name())),
        }
    }

    /// Human-readable rendering used by `print`.
    pub fn display(&self) -> String {
        match self {
            Value::Unit => "()".into(),
            Value::Int(v) => v.to_string(),
            Value::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 1e16 {
                    format!("{:.1}", v)
                } else {
                    format!("{}", v)
                }
            }
            Value::Bool(v) => v.to_string(),
            Value::Str(s) => s.clone(),
            Value::Tensor(t) => format_tensor(&t.borrow().data),
            Value::List(l) => {
                let items: Vec<String> = l.borrow().iter().map(|v| v.display()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Struct { name, fields } => {
                let items: Vec<String> =
                    fields.iter().map(|(n, v)| format!("{}: {}", n, v.display())).collect();
                format!("{}{{{}}}", name, items.join(", "))
            }
            Value::Layer(l) => format!("{}", l.describe()),
            Value::Optimizer(o) => format!("{}", o.borrow().describe()),
            Value::Dataset(d) => format!("Dataset(rows={})", d.len()),
            Value::Range(a, b) => format!("range({}, {})", a, b),
        }
    }
}

pub fn format_tensor(t: &Tensor) -> String {
    let mut vals = Vec::with_capacity(t.numel());
    for i in 0..t.numel() {
        let v = t.f64_at(i);
        vals.push(if v.fract() == 0.0 && v.abs() < 1e16 {
            format!("{:.1}", v)
        } else {
            format!("{}", v)
        });
    }
    let body = vals.join(", ");
    if t.shape.is_empty() {
        format!("Tensor({})", body)
    } else {
        format!("Tensor[{}]({})", t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "), body)
    }
}

/// Convert a nested Neura list into a rectangular tensor.
pub fn tensor_from_value(v: &Value, dtype: DType) -> Result<Tensor, String> {
    let mut vals = Vec::new();
    let shape = flatten_value(v, &mut vals)?;
    let t = Tensor::from_f64(dtype, shape, vals)?;
    Ok(t)
}

fn flatten_value(v: &Value, out: &mut Vec<f64>) -> Result<Vec<usize>, String> {
    match v {
        Value::List(l) => {
            let items = l.borrow();
            if items.is_empty() {
                return Ok(vec![0]);
            }
            let mut inner: Option<Vec<usize>> = None;
            for item in items.iter() {
                let s = flatten_value(item, out)?;
                match &inner {
                    Some(prev) if prev != &s => {
                        return Err(format!(
                            "ragged tensor literal: nested shapes {:?} and {:?}",
                            prev, s
                        ))
                    }
                    None => inner = Some(s),
                    _ => {}
                }
            }
            let mut shape = vec![items.len()];
            shape.extend(inner.unwrap_or_default());
            Ok(shape)
        }
        Value::Tensor(t) => {
            let b = t.borrow();
            out.extend((0..b.data.numel()).map(|i| b.data.f64_at(i)));
            Ok(b.data.shape.clone())
        }
        Value::Int(_) | Value::Float(_) | Value::Bool(_) => {
            out.push(v.as_f64()?);
            Ok(Vec::new())
        }
        other => Err(format!("cannot build a tensor from {}", other.type_name())),
    }
}
