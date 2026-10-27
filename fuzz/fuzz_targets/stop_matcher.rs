//! For any text, stop strings and split points, the matcher emits exactly the text before the
//! first stop to complete and never emits a stop string.
#![no_main]

use engine::stop::{StopMatcher, StopResult, first_stop};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (String, Vec<String>, Vec<u8>)| {
    let (text, stops, cuts) = input;
    let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).take(4).collect();
    let mut bounds: Vec<usize> = cuts
        .iter()
        .map(|&c| c as usize % (text.len() + 1))
        .filter(|&i| text.is_char_boundary(i))
        .collect();
    bounds.extend([0, text.len()]);
    bounds.sort_unstable();
    bounds.dedup();

    let mut m = StopMatcher::new(stops.clone());
    let mut out = String::new();
    let mut stopped = false;
    for w in bounds.windows(2) {
        match m.push(&text[w[0]..w[1]]) {
            StopResult::Continue(t) => out += &t,
            StopResult::Stopped(t) => {
                out += &t;
                stopped = true;
                break;
            }
        }
    }
    if !stopped {
        out += &m.flush();
    }
    match first_stop(&text, &stops) {
        Some(at) => assert!(stopped && out == text[..at]),
        None => assert!(!stopped && out == text),
    }
});
