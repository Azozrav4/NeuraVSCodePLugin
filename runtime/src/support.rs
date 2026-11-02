//! Runtime support: profiling, memory accounting, and artifact caches.

use neura_backends::tensor::Tensor;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default, Clone, Copy)]
pub struct Entry {
    pub calls: u64,
    pub nanos: u128,
}

#[derive(Default)]
pub struct Profiler {
    pub enabled: bool,
    entries: BTreeMap<String, Entry>,
}

impl Profiler {
    pub fn new(enabled: bool) -> Profiler {
        Profiler { enabled, entries: BTreeMap::new() }
    }

    pub fn record(&mut self, name: &str, nanos: u128) {
        if !self.enabled {
            return;
        }
        let e = self.entries.entry(name.to_string()).or_default();
        e.calls += 1;
        e.nanos += nanos;
    }

    pub fn entries(&self) -> Vec<(String, Entry)> {
        let mut v: Vec<(String, Entry)> = self.entries.iter().map(|(k, e)| (k.clone(), *e)).collect();
        v.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
        v
    }

    pub fn report(&self) -> String {
        let mut out = String::from("operation            calls      time(ms)   share\n");
        let total: u128 = self.entries.values().map(|e| e.nanos).sum();
        for (name, e) in self.entries() {
            let ms = e.nanos as f64 / 1e6;
            let share = if total == 0 { 0.0 } else { e.nanos as f64 * 100.0 / total as f64 };
            out.push_str(&format!("{:<20} {:>6} {:>12.3} {:>6.1}%\n", name, e.calls, ms, share));
        }
        out
    }

    pub fn reset(&mut self) {
        self.entries.clear();
    }
}

#[derive(Default, Clone, Copy)]
pub struct MemoryStats {
    pub allocations: u64,
    pub frees: u64,
    pub live_bytes: usize,
    pub peak_bytes: usize,
    pub pooled_reuses: u64,
}

/// Tracks bytes held by tensor registers. Slots are reused when a register is
/// overwritten or dropped, mirroring the compiler's buffer plan.
#[derive(Default)]
pub struct MemoryManager {
    slots: HashMap<u32, usize>,
    stats: MemoryStats,
}

impl MemoryManager {
    pub fn assign(&mut self, reg: u32, t: &Tensor) {
        let bytes = t.numel() * t.dtype.size();
        let previous = self.slots.insert(reg, bytes);
        if let Some(old) = previous {
            self.stats.live_bytes = self.stats.live_bytes.saturating_sub(old);
            self.stats.frees += 1;
        }
        self.stats.allocations += 1;
        self.stats.live_bytes += bytes;
        if self.stats.live_bytes > self.stats.peak_bytes {
            self.stats.peak_bytes = self.stats.live_bytes;
        }
    }

    pub fn release(&mut self, reg: u32) {
        if let Some(old) = self.slots.remove(&reg) {
            self.stats.live_bytes = self.stats.live_bytes.saturating_sub(old);
            self.stats.frees += 1;
        }
    }

    pub fn note_pool_reuse(&mut self, n: u64) {
        self.stats.pooled_reuses += n;
    }

    pub fn stats(&self) -> MemoryStats {
        self.stats
    }
}

pub struct CompilationCache {
    dir: PathBuf,
}

impl CompilationCache {
    pub fn new(base: impl AsRef<Path>) -> CompilationCache {
        CompilationCache { dir: base.as_ref().to_path_buf() }
    }

    pub fn default_location() -> CompilationCache {
        let base = std::env::var("NEURA_CACHE").unwrap_or_else(|_| ".neura-cache".to_string());
        CompilationCache::new(base)
    }

    fn path(&self, hash: u64, ext: &str) -> PathBuf {
        self.dir.join(format!("{:016x}.{}", hash, ext))
    }

    pub fn store(&self, hash: u64, artifact: &str) -> Result<PathBuf, String> {
        fs::create_dir_all(&self.dir)
            .map_err(|e| format!("cannot create cache directory {}: {}", self.dir.display(), e))?;
        let p = self.path(hash, "neurabc");
        fs::write(&p, artifact).map_err(|e| format!("cannot write artifact {}: {}", p.display(), e))?;
        Ok(p)
    }

    pub fn load(&self, hash: u64) -> Option<String> {
        fs::read_to_string(self.path(hash, "neurabc")).ok()
    }
}

pub struct AutotuneCache {
    entries: HashMap<String, (usize, u64)>,
}

impl Default for AutotuneCache {
    fn default() -> Self {
        AutotuneCache { entries: HashMap::new() }
    }
}

impl AutotuneCache {
    pub fn get(&self, key: &str) -> Option<(usize, u64)> {
        self.entries.get(key).copied()
    }

    pub fn put(&mut self, key: String, config: usize, nanos: u64) {
        self.entries.insert(key, (config, nanos));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
