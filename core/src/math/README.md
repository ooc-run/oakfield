# Math Helpers

`/core/src/math` contains special functions and numerical helpers used
by operators, KernelIR, and analysis code.

## Stable-Oriented Areas

- Airy and Bessel helpers.
- Classical digamma, trigamma, tetragamma, finite ladder, and related
  special-function utilities.
- Fourier helpers.
- Theta helpers.
- Riemann zeta and completed xi evaluators, plus complex log-gamma helpers used
  by their internals.

These areas have tests and benchmark coverage. Their public contracts
are intended to be more stable than the experimental q-method areas below.

## Experimental Areas

The following APIs are experimental:

- q-zeta: `qzeta.c`
- q-digamma: `qdigamma.c`
- q-number: `qnumber.c`
- q-ladder/q-phi helpers: `q_ladder.c`

Experimental means the q-method API names, result structs, error models, branch
selection, convergence behavior, supported input domains, and performance may
change. Existing tests, examples, benchmarks, and API docs should keep the
experimental label visible for q-method surfaces.

## Contributor Notes

- Prefer safe APIs with structured reports when exposing failure-prone numerical
  methods.
- Document input domains and convergence limits at the public declaration.
- Keep fast-math disabled for sources that rely on precise IEEE behavior,
  special values, or finite checks.
- Benchmarks for experimental methods should be labeled exploratory.
