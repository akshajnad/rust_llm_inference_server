//! Opt-in per-stage GPU timing for the forward pass (`RLIS_PROFILE=1`). Each span synchronizes
//! the device before and after, so it serializes the GPU and slows everything down: use it to
//! find where time goes, never for throughput numbers. Disabled, a span costs a cached flag check.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use candle_core::{Device, Result};

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RLIS_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

thread_local! {
    static TOTALS: RefCell<BTreeMap<&'static str, (Duration, u64)>> = const { RefCell::new(BTreeMap::new()) };
}

/// Runs `f`, attributing its GPU time to `name` when profiling is on.
pub fn span<T>(dev: &Device, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled() {
        return f();
    }
    dev.synchronize()?;
    let start = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    let took = start.elapsed();
    TOTALS.with(|t| {
        let mut totals = t.borrow_mut();
        let e = totals.entry(name).or_default();
        e.0 += took;
        e.1 += 1;
    });
    Ok(out)
}

/// Totals since the last call, largest first, and resets them.
pub fn take_report() -> Vec<(&'static str, Duration, u64)> {
    let mut v: Vec<_> = TOTALS
        .with(|t| std::mem::take(&mut *t.borrow_mut()))
        .into_iter()
        .map(|(k, (d, n))| (k, d, n))
        .collect();
    v.sort_by_key(|r| std::cmp::Reverse(r.1));
    v
}
