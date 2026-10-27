/// Finds stop strings in streamed text without ever emitting any part of one.
///
/// Text that could still turn out to be the start of a stop string is held back until the
/// next piece arrives decides it. The result never depends on how the text was split into
/// pieces (tokens).
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StopResult {
    /// Text safe to emit now; generation continues.
    Continue(String),
    /// A stop string matched; emit this text (everything before it) and finish.
    Stopped(String),
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            stops,
            pending: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> StopResult {
        if self.stops.is_empty() {
            return StopResult::Continue(text.to_string());
        }
        self.pending.push_str(text);
        if let Some(at) = first_stop(&self.pending, &self.stops) {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            return StopResult::Stopped(out);
        }
        let hold = self.held_suffix_len();
        let emit_to = self.pending.len() - hold;
        let out = self.pending[..emit_to].to_string();
        self.pending.drain(..emit_to);
        StopResult::Continue(out)
    }

    /// Releases held-back text when generation ends for another reason.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Byte length of the longest suffix of `pending` that is a proper prefix of a stop string.
    fn held_suffix_len(&self) -> usize {
        let p = &self.pending;
        p.char_indices()
            .map(|(i, _)| i)
            .find(|&i| self.stops.iter().any(|s| s.starts_with(&p[i..])))
            .map_or(0, |i| p.len() - i)
    }
}

/// Start of the stop occurrence that completes first: the earliest end position, with ties
/// going to the earlier start. Picking the earliest start instead would make the output
/// depend on token boundaries, because a long stop that starts first but completes later
/// never happens if a shorter stop completes, and ends generation, before it.
pub fn first_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|s| text.find(s.as_str()).map(|at| (at + s.len(), at)))
        .min()
        .map(|(_, at)| at)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for p in pieces {
            match m.push(p) {
                StopResult::Continue(t) => out += &t,
                StopResult::Stopped(t) => return (out + &t, true),
            }
        }
        (out + &m.flush(), false)
    }

    #[test]
    fn no_stops_passes_through() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn stop_split_across_pieces_is_not_emitted() {
        assert_eq!(
            run(&["\n\nQ:"], &["A: 4", "\n", "\nQ", ":", " next"]),
            ("A: 4".into(), true)
        );
    }

    #[test]
    fn false_start_is_released() {
        assert_eq!(run(&["END"], &["the E", "ND"]), ("the ".into(), true));
        assert_eq!(run(&["END"], &["the E", "nd"]), ("the End".into(), false));
    }

    #[test]
    fn first_completed_stop_wins() {
        // "b" completes before "abc" does, however the text is chunked.
        assert_eq!(run(&["b", "abc"], &["xxabc"]), ("xxa".into(), true));
        assert_eq!(run(&["b", "abc"], &["xxa", "bc"]), ("xxa".into(), true));
        // Both complete together: the earlier start wins.
        assert_eq!(run(&["c", "abc"], &["xxabc"]), ("xx".into(), true));
    }

    /// The case cargo-fuzz found: a long stop starting first, a short one completing first.
    #[test]
    fn long_stop_started_earlier_loses_to_short_stop_completed_earlier() {
        let stops = ["a b c d", "b"];
        assert_eq!(run(&stops, &["xa b c d"]), ("xa ".into(), true));
        assert_eq!(run(&stops, &["xa b", " c d"]), ("xa ".into(), true));
    }

    #[test]
    fn held_text_is_emitted_as_soon_as_possible() {
        let mut m = StopMatcher::new(vec!["ab".into()]);
        assert_eq!(m.push("xa"), StopResult::Continue("x".into()));
        assert_eq!(m.push("c"), StopResult::Continue("ac".into()));
    }

    #[test]
    fn multibyte_text_splits_on_char_boundaries() {
        assert_eq!(run(&["€x"], &["a€", "€x"]), ("a€".into(), true));
        assert_eq!(run(&["😀!"], &["héllo😀", "?"]), ("héllo😀?".into(), false));
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(4096))]

        /// However the text is split into pieces, the output is exactly the text up to the
        /// first stop to complete, and no stop string is ever emitted.
        #[test]
        fn output_is_text_before_first_stop(
            text in "[ab€\n]{0,40}",
            stops in proptest::collection::vec("[ab€\n]{1,3}", 0..3),
            cuts in proptest::collection::vec(0usize..48, 0..10),
        ) {
            let chars: Vec<char> = text.chars().collect();
            let mut bounds: Vec<usize> = cuts.into_iter().map(|c| c.min(chars.len())).collect();
            bounds.extend([0, chars.len()]);
            bounds.sort_unstable();
            bounds.dedup();
            let pieces: Vec<String> = bounds.windows(2).map(|w| chars[w[0]..w[1]].iter().collect()).collect();
            let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let stop_refs: Vec<&str> = stops.iter().map(String::as_str).collect();

            let (out, stopped) = run(&stop_refs, &piece_refs);
            match first_stop(&text, &stops) {
                Some(at) => {
                    proptest::prop_assert!(stopped);
                    proptest::prop_assert_eq!(out, &text[..at]);
                }
                None => {
                    proptest::prop_assert!(!stopped);
                    proptest::prop_assert_eq!(out, text);
                }
            }
        }
    }
}
