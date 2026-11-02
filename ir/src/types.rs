//! Core type and shape representation shared by every compiler layer.

use std::fmt;

/// Element type of a tensor (or a scalar).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }

    pub fn size(self) -> usize {
        match self {
            DType::Bool | DType::I8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I8 | DType::I32 | DType::I64)
    }

    pub fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "f32" | "float32" | "fp32" => DType::F32,
            "f64" | "float64" | "fp64" => DType::F64,
            "f16" | "float16" | "fp16" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            "i8" | "int8" => DType::I8,
            "i32" | "int32" => DType::I32,
            "i64" | "int64" => DType::I64,
            "bool" => DType::Bool,
            _ => return None,
        })
    }

    /// Dtype used when two element types are combined.
    pub fn promote(self, other: DType) -> DType {
        use DType::*;
        fn rank(d: DType) -> u8 {
            match d {
                Bool => 0,
                I8 => 1,
                I32 => 2,
                I64 => 3,
                F16 => 4,
                BF16 => 5,
                F32 => 6,
                F64 => 7,
            }
        }
        if rank(self) >= rank(other) {
            self
        } else {
            other
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A tensor dimension: statically known, symbolic, or not yet inferred.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    Static(usize),
    Symbol(String),
    Unknown,
}

impl Dim {
    pub fn concrete(&self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Dim::Unknown)
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(n) => write!(f, "{}", n),
            Dim::Symbol(s) => f.write_str(s),
            Dim::Unknown => f.write_str("?"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorTy {
    pub dtype: DType,
    pub shape: Vec<Dim>,
}

impl TensorTy {
    pub fn new(dtype: DType, shape: Vec<Dim>) -> Self {
        TensorTy { dtype, shape }
    }

    pub fn scalar(dtype: DType) -> Self {
        TensorTy { dtype, shape: vec![] }
    }

    pub fn static_shape(&self) -> Option<Vec<usize>> {
        self.shape.iter().map(|d| d.concrete()).collect()
    }

    pub fn numel(&self) -> Option<usize> {
        let dims = self.static_shape()?;
        Some(dims.iter().product())
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for TensorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[")?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "; {}]", self.dtype)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct FnSig {
    pub params: Vec<Ty>,
    pub ret: Box<Ty>,
}

/// Language-level types. `Opaque` is used for library-defined objects
/// (layers, optimizers, datasets, ...) so the core type system stays small.
#[derive(Clone, PartialEq, Debug)]
pub enum Ty {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    Tensor(TensorTy),
    List(Box<Ty>),
    Struct(String),
    Opaque(String),
    Module(String),
    Func(FnSig),
    Any,
    Never,
}

impl Ty {
    pub fn is_copy(&self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Bool | Ty::Unit)
    }

    pub fn tensor(&self) -> Option<&TensorTy> {
        match self {
            Ty::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Ty::Int => "Int".into(),
            Ty::Float => "Float".into(),
            Ty::Bool => "Bool".into(),
            Ty::Str => "Str".into(),
            Ty::Unit => "Unit".into(),
            Ty::Tensor(t) => t.to_string(),
            Ty::List(t) => format!("List[{}]", t.name()),
            Ty::Struct(s) => s.clone(),
            Ty::Opaque(s) => s.clone(),
            Ty::Module(s) => format!("module {}", s),
            Ty::Func(sig) => {
                let ps: Vec<String> = sig.params.iter().map(|p| p.name()).collect();
                format!("fn({}) -> {}", ps.join(", "), sig.ret.name())
            }
            Ty::Any => "Any".into(),
            Ty::Never => "Never".into(),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// Shape of a scalar (rank 0) f32 tensor.
pub fn scalar_tensor() -> Ty {
    Ty::Tensor(TensorTy::new(DType::F32, vec![]))
}
