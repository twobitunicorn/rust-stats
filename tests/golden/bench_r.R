# Time R's base-stats equivalents of the operations rust-stats benchmarks.
# Mirrors examples/bench.rs and tests/golden/bench_statsmodels.py.
#
# Run with:
#   Rscript tests/golden/bench_r.R

set.seed(0xCAFE)

# ── helpers ────────────────────────────────────────────────────────────

time_iters <- function(iters, fn) {
  fn()  # warmup
  samples <- numeric(iters)
  for (i in seq_len(iters)) {
    t0 <- Sys.time()
    fn()
    samples[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  median(samples)
}

report <- function(label, n, extra, secs) {
  cat(sprintf("%-22s n=%-6d %-20s %10.3f ms\n", label, n, extra, secs * 1e3))
}

series_with_seasonality <- function(n, period, seed = 0xCAFE) {
  set.seed(seed)
  i <- 0:(n - 1)
  trend <- 10.0 + 0.05 * i
  phase <- 2 * pi * (i %% period) / period
  seasonal <- 3.0 * sin(phase) + 1.5 * cos(2 * phase)
  trend + seasonal + rnorm(n) * 0.5
}

# ── LOESS ──────────────────────────────────────────────────────────────

bench_loess <- function() {
  # delta=0 forces a per-point fit; without it R uses an internal jump-like
  # approximation that does ~100 fits and interpolates the rest, which is
  # not what rust-stats and statsmodels (it=0) do.
  for (cfg in list(c(100, 100), c(1000, 30), c(5000, 10))) {
    n <- cfg[1]; iters <- cfg[2]
    set.seed(0xBEEF)
    y <- rnorm(n)
    x <- 0:(n - 1)
    secs <- time_iters(iters, function() {
      invisible(lowess(x, y, f = 0.3, iter = 0, delta = 0))
    })
    report("loess (deg=1)", n, "span=0.3", secs)
  }
}

# ── STL ────────────────────────────────────────────────────────────────

bench_stl <- function() {
  for (cfg in list(c(144, 12, 50), c(720, 12, 20), c(2880, 24, 10))) {
    n <- cfg[1]; period <- cfg[2]; iters <- cfg[3]
    y <- ts(series_with_seasonality(n, period), frequency = period)
    secs <- time_iters(iters, function() {
      invisible(stl(y, s.window = 7, robust = FALSE, inner = 2, outer = 0, s.jump = 1, t.jump = 1, l.jump = 1))
    })
    report("stl", n, sprintf("period=%d", period), secs)
  }
}

# ── seasonal_decompose ────────────────────────────────────────────────

bench_seasonal_decompose <- function() {
  for (cfg in list(c(144, 12, 200), c(720, 12, 100), c(2880, 24, 50))) {
    n <- cfg[1]; period <- cfg[2]; iters <- cfg[3]
    y <- ts(series_with_seasonality(n, period), frequency = period)
    secs_add <- time_iters(iters, function() {
      invisible(decompose(y, type = "additive"))
    })
    report("seasonal_decompose +", n, sprintf("period=%d", period), secs_add)
    # R's decompose requires strictly positive y for multiplicative; shift.
    y_pos <- y - min(y) + 1
    secs_mul <- time_iters(iters, function() {
      invisible(decompose(y_pos, type = "multiplicative"))
    })
    report("seasonal_decompose *", n, sprintf("period=%d", period), secs_mul)
  }
}

# ── Batched (50 series, R has no native batched form so loop) ─────────

bench_batched <- function() {
  set.seed(0xABCD)
  make_series_set <- function(n, p, period) {
    out <- matrix(0, nrow = p, ncol = n)
    for (j in seq_len(p)) {
      out[j, ] <- series_with_seasonality(n, period, seed = j)
    }
    out
  }

  for (cfg in list(c(1000, 50, 12, 20), c(720, 50, 12, 30), c(2880, 50, 24, 10))) {
    n <- cfg[1]; p <- cfg[2]; period <- cfg[3]; iters <- cfg[4]
    series <- make_series_set(n, p, period)
    secs <- time_iters(iters, function() {
      for (j in seq_len(p)) {
        invisible(stl(ts(series[j, ], frequency = period),
                      s.window = 7, robust = FALSE, inner = 2, outer = 0, s.jump = 1, t.jump = 1, l.jump = 1))
      }
    })
    report("stl_batch (loop)", n, sprintf("p=%d period=%d", p, period), secs)
  }

  for (cfg in list(c(1000, 50, 12, 20), c(720, 50, 12, 30), c(2880, 50, 24, 10))) {
    n <- cfg[1]; p <- cfg[2]; period <- cfg[3]; iters <- cfg[4]
    series <- make_series_set(n, p, period)
    secs <- time_iters(iters, function() {
      for (j in seq_len(p)) {
        invisible(decompose(ts(series[j, ], frequency = period), type = "additive"))
      }
    })
    report("seasonal_decompose_batch", n, sprintf("p=%d period=%d", p, period), secs)
  }

  for (cfg in list(c(1000, 50, 20), c(5000, 50, 5))) {
    n <- cfg[1]; p <- cfg[2]; iters <- cfg[3]
    series <- make_series_set(n, p, 12)
    x <- 0:(n - 1)
    secs <- time_iters(iters, function() {
      for (j in seq_len(p)) {
        invisible(lowess(x, series[j, ], f = 0.3, iter = 0, delta = 0))
      }
    })
    report("loess_batch (loop)", n, sprintf("p=%d span=0.3", p), secs)
  }
}

cat("# R benchmark\n\n")
bench_loess(); cat("\n")
bench_stl(); cat("\n")
bench_seasonal_decompose(); cat("\n")
bench_batched()
