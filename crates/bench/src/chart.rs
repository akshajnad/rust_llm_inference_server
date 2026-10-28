//! Minimal dependency-free SVG line charts for the README.

use std::fmt::Write;

pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    /// Optional text next to each point (e.g. the concurrency level).
    pub point_labels: Vec<String>,
}

pub struct Chart {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    /// Powers of two on the x axis (concurrency sweeps).
    pub log2_x: bool,
    pub series: Vec<Series>,
}

const W: f64 = 760.0;
const H: f64 = 460.0;
// Plot margins: left, right, top, bottom.
const L: f64 = 78.0;
const R: f64 = 24.0;
const T: f64 = 48.0;
const B: f64 = 58.0;
const COLORS: [&str; 6] = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#64748b",
];

/// Round tick values from `lo` up to the first one at or above `hi`, in steps of 1, 2 or 5
/// times a power of ten, so the last tick bounds the data.
pub fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if hi <= lo {
        return vec![lo];
    }
    let raw = (hi - lo) / target as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    let start = (lo / step).floor() * step;
    let end = (hi / step - 1e-9).ceil() * step;
    let n = ((end - start) / step).round() as usize;
    (0..=n).map(|i| start + i as f64 * step).collect()
}

fn fmt_num(v: f64) -> String {
    if v.abs() >= 1000.0 {
        format!("{:.0}", v)
    } else if v.fract().abs() < 1e-9 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

impl Chart {
    pub fn svg(&self) -> String {
        let pts = self.series.iter().flat_map(|s| s.points.iter());
        let (mut xmin, mut xmax, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY, 0f64);
        for &(x, y) in pts {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymax = ymax.max(y);
        }
        let tx = |x: f64| -> f64 { if self.log2_x { x.max(1e-9).log2() } else { x } };
        let (ax0, ax1) = if self.log2_x {
            (tx(xmin).floor(), tx(xmax).ceil())
        } else {
            (0f64.min(xmin), xmax)
        };
        let y_ticks = nice_ticks(0.0, ymax * 1.05, 6);
        let ay1 = *y_ticks.last().unwrap_or(&1.0);
        let px = |x: f64| L + (tx(x) - ax0) / (ax1 - ax0).max(1e-9) * (W - L - R);
        let py = |y: f64| H - B - y / ay1.max(1e-9) * (H - T - B);

        let mut s = String::new();
        let _ = write!(
            s,
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, sans-serif" font-size="12">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{L}" y="26" font-size="16" font-weight="600" fill="#0f172a">{}</text>
"##,
            esc(&self.title)
        );
        for y in &y_ticks {
            let yy = py(*y);
            let _ = writeln!(
                s,
                r##"<line x1="{L}" x2="{}" y1="{yy:.1}" y2="{yy:.1}" stroke="#e2e8f0"/><text x="{}" y="{:.1}" text-anchor="end" fill="#475569">{}</text>"##,
                W - R,
                L - 8.0,
                yy + 4.0,
                fmt_num(*y)
            );
        }
        let x_ticks: Vec<f64> = if self.log2_x {
            (ax0 as i32..=ax1 as i32).map(|e| 2f64.powi(e)).collect()
        } else {
            nice_ticks(ax0, ax1, 6)
        };
        for x in x_ticks {
            let xx = px(x);
            let _ = writeln!(
                s,
                r##"<line x1="{xx:.1}" x2="{xx:.1}" y1="{}" y2="{}" stroke="#f1f5f9"/><text x="{xx:.1}" y="{}" text-anchor="middle" fill="#475569">{}</text>"##,
                T,
                H - B,
                H - B + 18.0,
                fmt_num(x)
            );
        }
        let _ = writeln!(
            s,
            r##"<line x1="{L}" x2="{}" y1="{}" y2="{}" stroke="#94a3b8"/>"##,
            W - R,
            H - B,
            H - B
        );
        let _ = writeln!(
            s,
            r##"<text x="{}" y="{}" text-anchor="middle" fill="#334155">{}</text>"##,
            (L + W - R) / 2.0,
            H - 14.0,
            esc(&self.x_label)
        );
        let _ = writeln!(
            s,
            r##"<text transform="translate(18,{}) rotate(-90)" text-anchor="middle" fill="#334155">{}</text>"##,
            (T + H - B) / 2.0,
            esc(&self.y_label)
        );
        for (i, ser) in self.series.iter().enumerate() {
            let c = COLORS[i % COLORS.len()];
            let path: Vec<String> = ser
                .points
                .iter()
                .map(|&(x, y)| format!("{:.1},{:.1}", px(x), py(y)))
                .collect();
            let _ = writeln!(
                s,
                r#"<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{}"/>"#,
                path.join(" ")
            );
            for (j, &(x, y)) in ser.points.iter().enumerate() {
                let _ = writeln!(
                    s,
                    r#"<circle cx="{:.1}" cy="{:.1}" r="3.6" fill="{c}"/>"#,
                    px(x),
                    py(y)
                );
                if let Some(t) = ser.point_labels.get(j) {
                    let _ = writeln!(
                        s,
                        r#"<text x="{:.1}" y="{:.1}" font-size="10" fill="{c}">{}</text>"#,
                        px(x) + 6.0,
                        py(y) - 6.0,
                        esc(t)
                    );
                }
            }
            let ly = T + 12.0 + i as f64 * 18.0;
            let _ = writeln!(
                s,
                r##"<rect x="{}" y="{}" width="14" height="3" fill="{c}"/><text x="{}" y="{}" fill="#0f172a">{}</text>"##,
                L + 14.0,
                ly - 4.0,
                L + 34.0,
                ly,
                esc(&ser.label)
            );
        }
        s.push_str("</svg>\n");
        s
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_round_and_cover_the_range() {
        // The last tick bounds the data (the first version stopped below the maximum, so the
        // highest point was drawn above the plot).
        assert_eq!(
            nice_ticks(0.0, 1180.0, 6),
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0]
        );
        assert_eq!(nice_ticks(0.0, 9.5, 5), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(*nice_ticks(0.0, 587.0, 6).last().unwrap(), 600.0);
        assert_eq!(*nice_ticks(0.0, 600.0, 6).last().unwrap(), 600.0);
    }

    #[test]
    fn renders_every_series() {
        let chart = Chart {
            title: "t".into(),
            x_label: "x".into(),
            y_label: "y".into(),
            log2_x: true,
            series: vec![
                Series {
                    label: "a & b".into(),
                    points: vec![(1.0, 10.0), (4.0, 30.0)],
                    point_labels: vec!["1".into(), "4".into()],
                },
                Series {
                    label: "c".into(),
                    points: vec![(1.0, 5.0), (64.0, 50.0)],
                    point_labels: vec![],
                },
            ],
        };
        let svg = chart.svg();
        assert_eq!(svg.matches("<polyline").count(), 2);
        assert_eq!(
            svg.matches(r#"font-size="10""#).count(),
            2,
            "one label per labelled point"
        );
        assert!(svg.contains("a &amp; b"));
    }
}
