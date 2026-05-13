# Operators

Operators are the user-facing building blocks that mutate or observe fields
inside a `SimContext`. Most operators follow an add/config/update lifecycle:

- `sim_add_*_operator(...)` registers an operator and returns its index.
- `sim_*_config(...)` copies the normalized configuration from a registered
  operator.
- `sim_*_update(...)` replaces or renormalizes the configuration and invalidates
  affected scheduler state.

## Categories

- `advection`: gradients, divergence, curl, spatial derivatives, analytic warp.
- `common`: shared utility helpers for operator implementations.
- `coupling`: field mixing, masks, segmented sieve helpers, Metal-oriented mix.
- `diffusion`: laplacian, dispersion, fractional memory, linear dissipative and
  spectral operators.
- `measurement`: observers, phase features, sieve, convolution, sound
  observation.
- `neural`: neural inference and hybrid operators.
- `noise`: stochastic and Ornstein-Uhlenbeck sources.
- `nonlinear`: complex math, chaos maps, hysteretic and generic math operators.
- `reaction`: reaction/remainder helpers.
- `stimulus`: procedural stimuli and analytic field generators.
- `thermostat`: scalar regulation operators.
- `utility`: copy, scale, coordinate, phase rotation, FFT conversion, zeroing.

## Contributor Notes

- Put shared implementation helpers under `common` when several operator
  families use them.
- Public operator headers should describe configuration defaults, ownership,
  complex-field behavior, and whether the operator can emit symbolic KernelIR.
- Move roadmap TODOs into `/docs/roadmap.md` or issue tracking instead
  of leaving them in public headers.
- The zeta-plane stimulus is experimental and should stay marked as such in
  docs, examples, tests, and benchmarks.
- The SDR observer is experimental and should stay marked as such in schema
  text, public API docs, Lua examples, and roadmap items until the SDR backend
  and nonblocking capture path are production-ready.
