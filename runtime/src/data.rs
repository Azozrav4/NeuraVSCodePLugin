//! Dataset loading and batching.
//!
//! The loader reads numeric CSV files (or a directory containing `train.csv`,
//! `data.csv` or `dataset.csv`). The last column is the label; the remaining
//! columns are features. No Python data stack is involved.

use neura_backends::tensor::Tensor;
use neura_ir::types::DType;
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub struct Dataset {
    pub features: Tensor,
    pub labels: Tensor,
}

impl Dataset {
    pub fn new(features: Tensor, labels: Tensor) -> Result<Dataset, String> {
        if features.rank() != 2 {
            return Err(format!("features must be 2-D, found shape {:?}", features.shape));
        }
        if features.shape[0] != labels.numel() {
            return Err(format!(
                "dataset has {} rows but {} labels",
                features.shape[0],
                labels.numel()
            ));
        }
        Ok(Dataset { features, labels })
    }

    pub fn len(&self) -> usize {
        self.features.shape[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Dataset, String> {
        let path = path.as_ref();
        if path.is_dir() {
            for name in ["train.csv", "data.csv", "dataset.csv"] {
                let candidate = path.join(name);
                if candidate.exists() {
                    return Dataset::load_file(&candidate);
                }
            }
            return Err(format!(
                "dataset directory {} does not contain train.csv, data.csv or dataset.csv",
                path.display()
            ));
        }
        Dataset::load_file(path)
    }

    fn load_file(path: &Path) -> Result<Dataset, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("cannot read dataset {}: {}", path.display(), e))?;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut vals = Vec::new();
            let mut numeric = true;
            for cell in line.split(',') {
                let cell = cell.trim();
                match cell.parse::<f64>() {
                    Ok(v) => vals.push(v),
                    Err(_) => {
                        numeric = false;
                        break;
                    }
                }
            }
            if !numeric {
                if rows.is_empty() {
                    // Header row: ignored.
                    continue;
                }
                return Err(format!(
                    "{}:{} is not numeric; datasets must contain numeric columns",
                    path.display(),
                    lineno + 1
                ));
            }
            if !vals.is_empty() {
                rows.push(vals);
            }
        }
        if rows.is_empty() {
            return Err(format!("dataset {} contains no data rows", path.display()));
        }
        let width = rows[0].len();
        if width < 2 {
            return Err(format!(
                "dataset {} needs at least one feature column and one label column",
                path.display()
            ));
        }
        for (i, r) in rows.iter().enumerate() {
            if r.len() != width {
                return Err(format!(
                    "dataset {} row {} has {} columns, expected {}",
                    path.display(),
                    i + 1,
                    r.len(),
                    width
                ));
            }
        }
        let n = rows.len();
        let mut feats = Vec::with_capacity(n * (width - 1));
        let mut labels = Vec::with_capacity(n);
        for r in &rows {
            feats.extend_from_slice(&r[..width - 1]);
            labels.push(r[width - 1] as i64);
        }
        let features = Tensor::from_f64(DType::F32, vec![n, width - 1], feats)?;
        let labels = Tensor::from_i64(DType::I64, vec![n], labels)?;
        Dataset::new(features, labels)
    }

    /// Split into batches. Shuffling is deterministic for a given seed.
    pub fn batches(&self, batch_size: usize, seed: u64) -> Vec<(Tensor, Tensor)> {
        let batch_size = batch_size.max(1);
        let n = self.len();
        let order = permutation(n, seed);
        let cols = self.features.shape[1];
        let mut out = Vec::new();
        for chunk in order.chunks(batch_size) {
            let mut feats = Vec::with_capacity(chunk.len() * cols);
            let mut labels = Vec::with_capacity(chunk.len());
            for &i in chunk {
                feats.extend((0..cols).map(|c| self.features.f64_at(i * cols + c)));
                labels.push(self.labels.i64_at(i));
            }
            let x = Tensor::from_f64(DType::F32, vec![chunk.len(), cols], feats)
                .expect("batch features are well-formed");
            let y = Tensor::from_i64(DType::I64, vec![chunk.len()], labels)
                .expect("batch labels are well-formed");
            out.push((x, y));
        }
        out
    }

    pub fn shuffle(&self, seed: u64) -> Dataset {
        let order = permutation(self.len(), seed);
        let cols = self.features.shape[1];
        let mut feats = Vec::with_capacity(self.features.numel());
        let mut labels = Vec::with_capacity(self.len());
        for &i in &order {
            feats.extend((0..cols).map(|c| self.features.f64_at(i * cols + c)));
            labels.push(self.labels.i64_at(i));
        }
        Dataset {
            features: Tensor::from_f64(DType::F32, vec![order.len(), cols], feats)
                .expect("shuffled features are well-formed"),
            labels: Tensor::from_i64(DType::I64, vec![order.len()], labels)
                .expect("shuffled labels are well-formed"),
        }
    }

    /// Split by a fraction in (0, 1); the first part is returned first.
    pub fn split(&self, fraction: f64) -> Result<(Dataset, Dataset), String> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err("split fraction must be between 0 and 1".into());
        }
        let cut = ((self.len() as f64) * fraction).round() as usize;
        let cols = self.features.shape[1];
        let subset = |range: std::ops::Range<usize>| -> Dataset {
            let count = range.len();
            let mut feats = Vec::with_capacity(count * cols);
            let mut labels = Vec::with_capacity(count);
            for i in range {
                feats.extend((0..cols).map(|c| self.features.f64_at(i * cols + c)));
                labels.push(self.labels.i64_at(i));
            }
            Dataset {
                features: Tensor::from_f64(DType::F32, vec![count, cols], feats)
                    .expect("split features are well-formed"),
                labels: Tensor::from_i64(DType::I64, vec![count], labels)
                    .expect("split labels are well-formed"),
            }
        };
        Ok((subset(0..cut), subset(cut..self.len())))
    }
}

fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}
