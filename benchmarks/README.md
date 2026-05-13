# Benchmarks

Benchmarks are opt-in developer tools, not correctness tests. They report local
timings for comparing changes on the same machine and build configuration.

## Running

```sh
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DOAKFIELD_BUILD_BENCHMARKS=ON \
  -DOAKFIELD_BUILD_TESTS=OFF \
  -DOAKFIELD_BUILD_EXAMPLES=OFF \
  -DOAKFIELD_BUILD_DOCS=OFF
cmake --build build-bench --target \
  benchmark_core \
  benchmark_experimental_math \
  benchmark_special_functions \
  benchmark_diffusion \
  benchmark_integrators \
  benchmark_segmented_sieve
build-bench/bin/benchmark_core --elements=65536 --iterations=128
build-bench/bin/benchmark_experimental_math --iterations=32
build-bench/bin/benchmark_special_functions --iterations=128 --samples=256
build-bench/bin/benchmark_diffusion --iterations=16
build-bench/bin/benchmark_integrators --iterations=128 --shape=1024
build-bench/bin/benchmark_segmented_sieve 200000 32768
```

The runner prints compiler/build settings that affect timing, including
fast-math, OpenMP, and vDSP compile-time availability where visible to the
benchmark. Keep generated timing output out of source control unless a release
process intentionally versions it.

The special-function comparison benchmark always includes Oakfield rows and
auto-detects optional GSL, Boost.Math, and MPFR rows. Boost.Math detection checks
compiler defaults plus Homebrew include and Cellar locations. If Boost headers
live somewhere else, configure with
`-DOAKFIELD_BENCH_BOOST_INCLUDE_DIR=/path/to/include` or the `boost/` directory
itself. It prints a grouped table by default; pass `--format=kv` for key-value
rows that are easier to parse from scripts. In table mode, the fastest row is
highlighted in green and the most accurate non-oracle row is highlighted in
cyan when color is enabled; use `--color=auto|always|never` to control ANSI
output. Real-valued
accuracy deltas use MPFR oracles when MPFR is available, including MPFR-derived
digamma finite differences for trigamma and tetragamma. Complex rows use the
best local Oakfield reference named in each section. Digamma and trigamma
include Oakfield adaptive-tail rows; pass `--adaptive-tol=EPS` to choose the
adaptive tail tolerance. The suite also includes complex
digamma/trigamma/tetragamma rows plus real and complex zeta/xi rows when
`OAKFIELD_ENABLE_ZETA_CORE=ON`.

Experimental backend code can be compiled into the same benchmark build with
`-DOAKFIELD_ENABLE_CUDA=ON` when the CUDA Toolkit, driver library, and NVRTC are
available, and with `-DOAKFIELD_ENABLE_METAL=ON` on Apple platforms with Metal.
Enable both switches only on hosts that provide both toolchains.

For before/after comparisons, keep the machine, compiler, build type,
fast-math setting, problem size, iteration count, and backend options fixed.
Run each benchmark a few times and compare medians rather than single samples.
CI benchmark lanes should be smoke checks for crashes and obvious setup
regressions, not performance threshold gates.

## Initial Coverage

- Field allocation and destruction.
- Non-owning field wrapping.
- Real-to-complex field promotion.
- Utility copy and scale operators through a prepared `SimContext` plan.
- Stable special-function calls.
- Stable special-function throughput against optional GSL, Boost.Math, and MPFR
  baselines where those libraries are available at configure time.
- CPU KernelIR pointwise launch throughput for a simple affine expression.
- Euler and RK4 real-field decay stepping.
- Exploratory q-method timings in `benchmark_experimental_math`.
- `benchmark_experimental_math` also reports Zeta/Xi timings when
  `OAKFIELD_ENABLE_ZETA_CORE=ON`.
- Legacy diffusion integration throughput and validation.
- Legacy integrator factory throughput across fixed/adaptive modes.
- Legacy segmented-sieve operator throughput.

Zeta/Xi and q-method benchmarks remain grouped in the same opt-in executable.
Only q-method rows use `EXPERIMENTAL` result names until their public accuracy
and performance contracts are stable.
