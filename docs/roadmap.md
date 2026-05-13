# Roadmap Notes

This file holds roadmap items that should not live as TODO comments in public
headers.

## Complex-Field Operator Modes

- Add magnitude-only or phase-only sinusoidal stimulus modes while preserving
  the other component.
- Add magnitude-only envelope behavior for complex Gaussian stimuli.
- Investigate phased sieve kernels that attenuate phase noise while preserving
  complex magnitude.
- Add configurable complex filtering modes for sieve operators, including
  component-wise and magnitude-oriented behavior.
- Add phase-aware mixer crossfades that preserve instantaneous magnitude/phase
  relationships.
- Add magnitude/phase selective kernels for fractional memory.

## Math Coverage

- Broaden domain tests for Zeta/Xi and q-zeta, q-digamma, q-number, and
  q-ladder.
- Add branch-selection tests for Zeta/Xi methods.
- Add convergence and failure-mode tests for q-methods.
- Broaden benchmark cases that are explicitly labeled exploratory until
  accuracy and stability expectations are finalized.

## Documentation And Packaging

- Generate API docs from public docblocks.
- Add examples that compile in CI.
- Add install/export support and a public include tree.
- Add contribution, license, and issue-template scaffolding.
