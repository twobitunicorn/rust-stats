# Time R's stats::arima() on the rust-stats ARIMA bench workloads.
#
# Pair with examples/bench_arima.rs and tests/golden/bench_arima_statsmodels.py
# for a three-way comparison.
#
# R's stats::arima defaults to method = "CSS-ML": CSS for starting
# values, then exact MLE refinement via Kalman filter. That's the
# closest analogue to rust-stats' ArimaMethod::CssMle.
#
# Run with:
#
#   Rscript tests/golden/bench_arima_r.R

time_iters <- function(iters, fn) {
  fn()
  s <- numeric(iters)
  for (i in seq_len(iters)) {
    t0 <- Sys.time()
    fn()
    s[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  median(s)
}

report <- function(label, n, extra, secs) {
  cat(sprintf("%-32s n=%-6d %-20s %10.2f ms\n", label, n, extra, secs * 1e3))
}

simulate_arma <- function(n, phi, theta, sigma, seed) {
  set.seed(seed)
  burn <- 200
  eps <- sigma * rnorm(n + burn)
  y <- numeric(n + burn)
  p <- length(phi); q <- length(theta)
  for (t in seq_along(y)) {
    yt <- eps[t]
    if (p > 0) for (i in seq_len(min(p, t - 1))) yt <- yt + phi[i] * y[t - i]
    if (q > 0) for (i in seq_len(min(q, t - 1))) yt <- yt + theta[i] * eps[t - i]
    y[t] <- yt
  }
  tail(y, n)
}

integrate_once <- function(y, start = 100) {
  cumsum(y) + start
}

bench_one <- function(label, y, order, iters, seasonal = list(order = c(0, 0, 0), period = 1)) {
  fn <- function() {
    tryCatch(
      invisible(stats::arima(y, order = order, seasonal = seasonal,
                             optim.control = list(maxit = 200))),
      error = function(e) NULL
    )
  }
  secs <- time_iters(iters, fn)
  report(paste0(label, " (R arima)"), length(y), "", secs)
}

bench_ar1 <- function() {
  for (cfg in list(c(144, 50), c(720, 20), c(2880, 5))) {
    n <- cfg[1]; iters <- cfg[2]
    y <- simulate_arma(n, 0.6, c(), 1, 0xA1)
    bench_one("ARIMA(1,0,0)", y, c(1, 0, 0), iters)
  }
}

bench_ma1 <- function() {
  for (cfg in list(c(144, 50), c(720, 20), c(2880, 5))) {
    n <- cfg[1]; iters <- cfg[2]
    y <- simulate_arma(n, c(), 0.5, 1, 0xA2)
    bench_one("ARIMA(0,0,1)", y, c(0, 0, 1), iters)
  }
}

bench_arma11 <- function() {
  for (cfg in list(c(144, 30), c(720, 15), c(2880, 3))) {
    n <- cfg[1]; iters <- cfg[2]
    y <- simulate_arma(n, 0.5, 0.3, 1, 0xA3)
    bench_one("ARIMA(1,0,1)", y, c(1, 0, 1), iters)
  }
}

bench_ima11 <- function() {
  for (cfg in list(c(144, 30), c(720, 15), c(2880, 3))) {
    n <- cfg[1]; iters <- cfg[2]
    arma <- simulate_arma(n, c(), -0.4, 1, 0xA4)
    y <- integrate_once(arma)
    bench_one("ARIMA(0,1,1)", y, c(0, 1, 1), iters)
  }
}

bench_arima111 <- function() {
  for (cfg in list(c(144, 20), c(720, 10), c(2880, 3))) {
    n <- cfg[1]; iters <- cfg[2]
    arma <- simulate_arma(n, 0.5, -0.3, 1, 0xA5)
    y <- integrate_once(arma)
    bench_one("ARIMA(1,1,1)", y, c(1, 1, 1), iters)
  }
}

bench_sarima_airline <- function() {
  for (cfg in list(c(144, 5), c(288, 3))) {
    n <- cfg[1]; iters <- cfg[2]
    arma <- simulate_arma(n, c(), -0.4, 1, 0xA6)
    i <- 0:(n - 1)
    trend <- 0.05 * i
    phase <- 2 * pi * (i %% 12) / 12
    seasonal <- 3 * sin(phase)
    y <- arma + trend + seasonal + 100
    for (k in 2:n) y[k] <- y[k] + y[k - 1] * 0.001
    bench_one("SARIMA(0,1,1)(0,1,1)[12]", y, c(0, 1, 1), iters,
              seasonal = list(order = c(0, 1, 1), period = 12))
  }
}

cat("# R bench: stats::arima() with method = CSS-ML (default)\n\n")
bench_ar1(); cat("\n")
bench_ma1(); cat("\n")
bench_arma11(); cat("\n")
bench_ima11(); cat("\n")
bench_arima111(); cat("\n")
bench_sarima_airline()
