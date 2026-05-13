# Core

`/core` contains the engine primitives used by the rest of the
package: fields, buffers, contexts, operators, KernelIR, runtime
state, math helpers, and diagnostics.

## Main Concepts

- Fields own or wrap multidimensional scalar storage. Start with
  `src/field.h`.
- Buffers provide smaller typed allocations for scalar data outside a full
  field. See `src/sim_buffer.h`.
- Contexts own the fields and operators that participate in a simulation. See
  `src/sim_context.h`.
- Operators describe work that can be scheduled, split, fused, or evaluated
  through KernelIR. See `src/operator.h` and `src/operator_split.h`.
- KernelIR is the intermediate representation used by backends for fused
  expression evaluation. See `src/kernel_ir.h`.
- Runtime state tracks execution metadata, profiling, scheduler state, and
  diagnostics.

## Reader Path

1. Read `src/field.h` and `src/field.c`.
2. Read `src/sim_context.h` to see how fields and operators are owned.
3. Read `src/operator.h` for operator metadata and contracts.
4. Read `src/operator_split.h` for the split-step operator interface.
5. Read `src/kernel_ir.h` only after the field/operator lifecycle is clear.

## Public Versus Internal

Public headers live under `../include/oakfield` and are the only include paths
new examples and downstream users should rely on. Implementation-only headers
live under explicit `internal/` source directories, including
`src/internal/kernel_ir_internal.h`, `src/internal/split_internal.h`, and
`src/math/internal/*_internal.h`.
