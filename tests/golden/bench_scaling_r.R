# Scaling sweep: R's stats::arima(method = "CSS-ML") on ARIMA(1, 1, 1)
# at n = 10^4, 10^5, 10^6, 10^7. Pair with examples/bench_scaling.rs.
#
# The largest size will take a while — R's `arima` is mature C/Fortran
# but it still does Kalman MLE per fit, so n = 10^7 is several minutes.
#
# Run with:
#
#   Rscript tests/golden/bench_scaling_r.R

set.seed(0x5CA1ED)

simulate_arima_111 <- function(n) {
  phi <- 0.5; theta <- -0.3
  eps <- rnorm(n)
  diff <- numeric(n)
  for (t in 2:n) {
    diff[t] <- 0.1 + phi * diff[t - 1] + theta * eps[t - 1] + eps[t]
  }
  y <- numeric(n)
  y[1] <- 100
  for (t in 2:n) y[t] <- y[t - 1] + diff[t]
  y
}

cat("# scaling sweep: stats::arima(method = \"CSS-ML\") — one fit per cell\n\n")
cat("  n              time          throughput\n")

for (n in c(1e4, 1e5, 1e6, 1e7)) {
  n <- as.integer(n)
  y <- simulate_arima_111(n)
  t0 <- Sys.time()
  fit <- tryCatch(
    suppressWarnings(stats::arima(y, order = c(1, 1, 1))),
    error = function(e) NULL
  )
  secs <- as.numeric(Sys.time() - t0, units = "secs")
  rate <- secs * 1e6 / n
  status <- if (is.null(fit)) "  (errored)" else ""
  cat(sprintf("  n=%-10d    %10.3f s    %7.2f us/pt%s\n", n, secs, rate, status))
  flush.console()
}
