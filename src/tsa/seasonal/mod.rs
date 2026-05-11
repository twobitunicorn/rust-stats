//! Seasonal-trend decomposition: Cleveland 1990 STL and the classical
//! moving-average `seasonal_decompose`. Both produce identically-shaped
//! `Decomposition` output.

pub mod decompose;
pub mod stl;

pub use decompose::seasonal_decompose;
pub use stl::stl;

/// Output of a seasonal-trend decomposition. The components reconstruct
/// the input where defined:
///
/// ```text
/// additive:        y[i] = trend[i] + seasonal[i] + residual[i]
/// multiplicative:  y[i] = trend[i] * seasonal[i] * residual[i]
/// ```
///
/// **Units note for multiplicative mode.** Trend is in the original
/// units (and is the *geometric* trend of `y`). Seasonal and residual
/// are **dimensionless ratios** centred around 1 — e.g. a seasonal value
/// of `1.40` means "this phase is 40% above the trend" and a residual of
/// `0.95` means "5% below the model's prediction". This matches R and
/// statsmodels. See `DecomposeMode::Multiplicative` for the rationale
/// and the implementation choice.
///
/// STL produces finite values everywhere; classical `seasonal_decompose`
/// has `NaN` at the first/last `period/2` positions where the centered
/// moving average can't be computed. When the input was processed with
/// `Missing::Interpolate`, `residual[i]` is also `NaN` at positions
/// where the input was originally non-finite.
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

/// Decomposition model: how the trend, seasonal, and residual components
/// combine to reconstruct the input.
///
/// `Additive` (`y = T + S + R`): seasonal amplitude is independent of
/// the level. Use when the seasonal swing is the same size regardless
/// of how high or low the trend is.
///
/// `Multiplicative` (`y = T · S · R`): seasonal amplitude scales with
/// the level. Use when the seasonal pattern is *proportional* — early
/// AirPassengers years have small seasonal wiggles, late years have
/// large ones, but the *ratio* of December-to-baseline is roughly
/// constant.
///
/// ## Implementation
///
/// For **STL**, multiplicative mode is implemented as `log → additive
/// STL → exp`: the input is log-transformed, additive STL runs on the
/// log series, and each output component is exponentiated. This is the
/// canonical workflow in the statistics literature; R's `stl()` and
/// statsmodels' `STL` both expect users to do this transformation
/// themselves. rust-stats bakes it in. Consequences:
///
/// - Requires **strictly positive** `y`; zero or negative values return
///   `NonPositiveForMultiplicative`.
/// - The trend is the **geometric trend** of `y` (in original units).
/// - The residual is a **dimensionless ratio** (≈ 1 means no anomaly).
/// - All other options compose normally — robust outer loop, jumps,
///   missing-data interpolation, and `Periodic` all run in log space and
///   the exp comes out the other side.
///
/// For **classical `seasonal_decompose`**, multiplicative mode uses the
/// direct algorithm (centered arithmetic MA, then `y / trend`, then
/// per-phase means normalised so the pattern's arithmetic mean across
/// one period is 1). No log transform. Matches statsmodels bitwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposeMode {
    Additive,
    Multiplicative,
}

/// How to smooth the seasonal cycle-subseries.
///
/// `Window(n)` (the standard Cleveland 1990 setup) fits a LOESS of the
/// given odd span (`n >= 7`) within each phase's subseries, letting the
/// seasonal pattern evolve smoothly over time.
///
/// `Periodic` constrains the seasonal pattern to be exactly periodic:
/// each phase's seasonal value is set to the (robustness-weighted) mean
/// of that phase's observations, repeated for every cycle. This matches
/// R's `stl(s.window = "periodic")` and is appropriate when the
/// seasonality is known to be stationary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalWindow {
    Window(u32),
    Periodic,
}

/// Policy for handling non-finite (NaN / ±Inf) entries in the input.
///
/// `Error` (default) preserves the existing behaviour: any non-finite
/// value returns `NonFinite`. `Interpolate` fills non-finite entries via
/// linear interpolation between adjacent finite values (leading/trailing
/// runs are filled with the nearest finite value), runs the decomposition
/// on the filled series, and propagates `NaN` back into the residual at
/// originally-missing positions. The trend and seasonal stay finite
/// everywhere — they are the model's estimates at the imputed points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Missing {
    #[default]
    Error,
    Interpolate,
}

/// Linearly interpolate non-finite values in `y`. Leading and trailing
/// runs of non-finite values are filled with the nearest finite value;
/// interior runs are filled by linear interpolation between the
/// surrounding finite endpoints. Returns `None` if `y` has no finite
/// values.
pub(crate) fn interpolate_missing(y: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let first = y.iter().position(|v| v.is_finite())?;
    let last = y.iter().rposition(|v| v.is_finite()).expect("first exists ⇒ last exists");

    let mut out = y.to_vec();
    let lead = out[first];
    for slot in out.iter_mut().take(first) {
        *slot = lead;
    }
    let tail = out[last];
    for slot in out.iter_mut().take(n).skip(last + 1) {
        *slot = tail;
    }

    let mut i = first + 1;
    while i < last {
        if out[i].is_finite() {
            i += 1;
            continue;
        }
        // Find next finite index j > i (must exist before `last` since
        // out[last] is finite).
        let mut j = i + 1;
        while !out[j].is_finite() {
            j += 1;
        }
        let lo = out[i - 1];
        let hi = out[j];
        let gap = (j - (i - 1)) as f64;
        for k in i..j {
            let alpha = (k - (i - 1)) as f64 / gap;
            out[k] = lo + alpha * (hi - lo);
        }
        i = j + 1;
    }
    Some(out)
}

/// Options for `stl`. Construct via `StlOpts::new(period)` for Cleveland
/// defaults and override individual fields with struct-update syntax.
#[derive(Debug, Clone)]
pub struct StlOpts {
    pub period: u32,
    /// Cycle-subseries smoothing policy. `Window(n)` (default) uses a
    /// LOESS of the given odd span (`n >= 7`); `Periodic` forces an
    /// exactly-repeating seasonal pattern (per-phase mean).
    pub seasonal_window: SeasonalWindow,
    /// LOESS span for the trend smoother. `None` uses Cleveland's
    /// recommended formula: smallest odd >=
    /// `1.5 * period / (1 - 1.5 / seasonal_window)`.
    pub trend_window: Option<u32>,
    /// Number of inner-loop iterations. Cleveland recommends 2.
    pub inner_iters: u32,
    /// Number of robust outer-loop iterations (Cleveland 1990 §3.5).
    /// `0` (default) skips the outer loop and produces a non-robust
    /// decomposition. `15` matches the default used by R's stl(robust=TRUE)
    /// and statsmodels' STL(robust=True). Each outer iteration recomputes
    /// per-point robustness weights from the previous residuals using the
    /// bisquare (Tukey biweight) function, then re-runs the inner loop with
    /// those weights folded into the cycle-subseries and trend LOESS fits.
    pub outer_iters: u32,
    pub mode: DecomposeMode,
    /// Cleveland 1990 "jump" parameter for the cycle-subseries smoother:
    /// fit LOESS at every `seasonal_jump`-th point in each subseries and
    /// linearly interpolate between them. Must be `>= 1`; `1` is exact.
    /// Cleveland recommends `round(period / 10)` for typical workloads.
    pub seasonal_jump: u32,
    /// Jump parameter for the trend LOESS. Must be `>= 1`; `1` is exact.
    pub trend_jump: u32,
    /// Jump parameter for the low-pass LOESS. Must be `>= 1`; `1` is exact.
    pub low_pass_jump: u32,
    /// How to handle non-finite values in the input. Defaults to `Error`.
    pub missing: Missing,
}

impl StlOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            seasonal_window: SeasonalWindow::Window(7),
            trend_window: None,
            inner_iters: 2,
            outer_iters: 0,
            mode: DecomposeMode::Additive,
            seasonal_jump: 1,
            trend_jump: 1,
            low_pass_jump: 1,
            missing: Missing::Error,
        }
    }
}

/// Options for `seasonal_decompose`. Construct via
/// `SeasonalDecomposeOpts::new(period)` for additive defaults.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposeOpts {
    pub period: u32,
    pub mode: DecomposeMode,
    /// How to handle non-finite values in the input. Defaults to `Error`.
    pub missing: Missing,
}

impl SeasonalDecomposeOpts {
    pub fn new(period: u32) -> Self {
        Self {
            period,
            mode: DecomposeMode::Additive,
            missing: Missing::Error,
        }
    }
}
