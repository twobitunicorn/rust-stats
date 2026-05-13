# Canonical catch22 C kernel benchmark

Times the upstream
[catch22 C kernel](https://github.com/DynamicsAndNeuralSystems/catch22)
on the same Gaussian inputs and `n` schedule as
`examples/bench_catch22.rs`. Pair the two outputs for an apples-to-apples
kernel-vs-kernel comparison (no Python wrapper in either column).

## Build & run

```sh
# 1. Clone the canonical C tree (anywhere outside this repo's src/).
git clone --depth 1 https://github.com/DynamicsAndNeuralSystems/catch22 \
    .reference/catch22-c

# 2. Build the C library.
cd .reference/catch22-c/C
gcc -O3 -c $(ls *.c | grep -v '^main.c$')
ar rcs libcatch22.a *.o

# 3. Build and run the bench harness against it.
gcc -O3 -Wno-format \
    -o bench_c \
    ../../../benchmarks/catch22_c/bench_c.c libcatch22.a -lm
./bench_c
```

The harness uses the same xorshift64 + Box–Muller RNG as the Rust bench,
so the inputs are statistically identical. It runs the full 22-feature
panel (z-score + every kernel main.c invokes) per timed iteration and
reports the median wall-clock per call.

`.reference/` is gitignored — keep the upstream clone there to avoid
mixing it into the rust-stats source tree.
