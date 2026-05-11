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

# ── center  ↔  scale(x, scale = FALSE) ────────────────────────────────

bench_center <- function() {
  for (cfg in list(c(10000, 200), c(100000, 100), c(1000000, 30))) {
    n <- cfg[1]; iters <- cfg[2]
    set.seed(0xC1)
    y <- rnorm(n)
    secs <- time_iters(iters, function() {
      invisible(scale(y, center = TRUE, scale = FALSE))
    })
    report("center (scale)", n, "", secs)
  }
}

# ── z_score  ↔  scale(x)  (ddof = 1; same as rust-stats) ──────────────

bench_z_score <- function() {
  for (cfg in list(c(10000, 200), c(100000, 100), c(1000000, 30))) {
    n <- cfg[1]; iters <- cfg[2]
    set.seed(0xC2)
    y <- rnorm(n)
    secs <- time_iters(iters, function() {
      invisible(scale(y))
    })
    report("z_score (scale)", n, "", secs)
  }
}

# ── min_max_scale  ↔  (x - min(x)) / diff(range(x)) ───────────────────

bench_min_max <- function() {
  for (cfg in list(c(10000, 200), c(100000, 100), c(1000000, 30))) {
    n <- cfg[1]; iters <- cfg[2]
    set.seed(0xC3)
    y <- rnorm(n)
    secs <- time_iters(iters, function() {
      rng <- range(y)
      invisible((y - rng[1]) / (rng[2] - rng[1]))
    })
    report("min_max_scale", n, "", secs)
  }
}

# ── box_cox  ↔  forecast::BoxCox if installed, else closed-form ───────

bench_box_cox <- function() {
  use_forecast <- requireNamespace("forecast", quietly = TRUE)
  for (cfg in list(c(10000, 100), c(100000, 30), c(1000000, 5))) {
    n <- cfg[1]; iters <- cfg[2]
    set.seed(0xC4)
    y <- exp(rnorm(n)) + 0.5  # strictly positive
    for (lmbda in c(0.0, 0.5, 2.0)) {
      fn <- if (use_forecast) {
        function() invisible(forecast::BoxCox(y, lambda = lmbda))
      } else {
        function() {
          if (lmbda == 0.0) invisible(log(y))
          else invisible((y^lmbda - 1) / lmbda)
        }
      }
      secs <- time_iters(iters, fn)
      report("box_cox", n, sprintf("lambda=%g", lmbda), secs)
    }
  }
}

# ── HoltWinters  ↔  stats::HoltWinters(x, alpha, beta, gamma) ─────────
#
# R's HoltWinters() requires a ts() object and seeds level/trend by
# regression on the first two seasons; we fix the smoothing constants so
# it doesn't optimize. SES and Holt's linear are spelled with
# `gamma=FALSE` / `beta=FALSE` respectively.

bench_holt_winters <- function() {
  for (cfg in list(c(144, 12, 200), c(720, 12, 100), c(2880, 24, 30))) {
    n <- cfg[1]; period <- cfg[2]; iters <- cfg[3]
    y_raw <- series_with_seasonality(n, period, seed = 0xC5)
    y_pos <- abs(y_raw) + 1  # strictly positive for multiplicative

    # SES — needs n > 2*period for HoltWinters(); fall back to closed-form
    # SES loop when too short. (Tiny series like n=144 with period=12 are
    # fine, but the API still demands period >= 2.)
    y_ts <- ts(y_raw, frequency = period)
    y_ts_pos <- ts(y_pos, frequency = period)

    secs <- time_iters(iters, function() {
      invisible(HoltWinters(y_ts, alpha = 0.5, beta = FALSE, gamma = FALSE))
    })
    report("hw SES", n, sprintf("period=%d", period), secs)

    secs <- time_iters(iters, function() {
      invisible(HoltWinters(y_ts, alpha = 0.5, beta = 0.1, gamma = FALSE))
    })
    report("hw Holt linear", n, sprintf("period=%d", period), secs)

    secs <- time_iters(iters, function() {
      invisible(HoltWinters(y_ts, alpha = 0.5, beta = 0.1, gamma = 0.2,
                            seasonal = "additive"))
    })
    report("hw additive", n, sprintf("period=%d", period), secs)

    secs <- time_iters(iters, function() {
      invisible(HoltWinters(y_ts_pos, alpha = 0.5, beta = 0.1, gamma = 0.2,
                            seasonal = "multiplicative"))
    })
    report("hw multiplicative", n, sprintf("period=%d", period), secs)
  }
}

cat("# R benchmark\n\n")
bench_loess(); cat("\n")
bench_stl(); cat("\n")
bench_seasonal_decompose(); cat("\n")
bench_batched(); cat("\n")
bench_center(); cat("\n")
bench_z_score(); cat("\n")
bench_min_max(); cat("\n")
bench_box_cox(); cat("\n")
bench_holt_winters()
