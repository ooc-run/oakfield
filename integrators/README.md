# Integrators

`/integrators` contains timestepper APIs and built-in schemes used to
advance fields from a drift callback or context-backed plan.

## Entry Points

- `src/integrator.h`: shared configuration, callbacks, workspace management,
  diagnostics, and step function contracts.
- `src/integrator_registry.h`: runtime lookup and factory registration for
  built-in and custom integrators.
- `src/integrator.c`: shared workspace, stochastic, timestep, and context-drift
  helpers.

## Built-In Schemes

- `euler`: first-order explicit stepper.
- `heun`: predictor-corrector stepper with an embedded estimate.
- `rk4`: classic fourth-order Runge-Kutta.
- `rkf45`: adaptive Runge-Kutta-Fehlberg.
- `backward_euler`: implicit-style backward Euler path.
- `crank_nicolson`: trapezoidal/Crank-Nicolson style stepper.
- `etdrk4`: exponential time-differencing RK4 for compatible linear/nonlinear
  splits.
- `subordination`: subordinated semigroup approximation.

## Contributor Notes

- Keep drift callbacks side-effect aware. The integrator may evaluate candidate
  states without committing them to the owning context.
- Document whether a scheme supports complex fields, stochastic increments,
  adaptive timesteps, and context-backed drift.
- Preserve deterministic seeds in tests and examples.
- Avoid adding new fields to `Integrator` without documenting ownership,
  lifecycle, and diagnostics impact.
