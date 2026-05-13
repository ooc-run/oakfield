# Approximation Tools

This directory holds opt-in developer tooling for generating special-function
approximants. The first tool, `oakfield_minimax`, is an MPFR-backed
Remez-style candidate generator for Chebyshev-basis polynomial and rational
approximations.

These tools are intentionally not part of normal builds. Configure them with:

```sh
cmake -S . -B build-approx \
  -DOAKFIELD_BUILD_TESTS=OFF \
  -DOAKFIELD_BUILD_EXAMPLES=OFF \
  -DOAKFIELD_BUILD_APPROX_TOOLS=ON
cmake --build build-approx --target oakfield_minimax
```

Example digamma candidate over Boost's central interval:

```sh
build-approx/bin/oakfield_minimax \
  --function=digamma \
  --a=1 --b=2 \
  --num-degree=5 --den-degree=6 \
  --grid=8192 --iterations=8 \
  --format=c
```

Supported oracle functions are currently:

- `digamma`
- `airy_ai`
- `bessel_jn` with `--order=N`

The generator emits candidates, not final production code. Before moving
coefficients into `core/src/math`, validate them with MPFR reference tests and
the special-function benchmark on the target intervals, including behavior near
zeros, poles, sign changes, and interval boundaries.
