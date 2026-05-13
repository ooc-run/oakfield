# Examples

These programs are small C entry points that mirror the workflows used by the
Lua scripts and runtime bridge: create fields, register operators, step
integrators, inspect results, and call math helpers directly.

## Running

```sh
cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DOAKFIELD_BUILD_EXAMPLES=ON
cmake --build /build --target example_minimal_field example_cpu_backend_kernel example_operator_context
/build/bin/example_minimal_field
```

Examples are enabled by default and write executables to the build tree's
`bin/` directory. They include public `oakfield/...` headers rather than
private source-tree paths.

## Programs

- `minimal_field.c`: create, inspect, fill, index, promote, and destroy a field.
- `cpu_backend_kernel.c`: build a small KernelIR affine expression and launch it
  on the CPU backend.
- `operator_context.c`: mirror Lua's `create/add_field/add_operator/step`
  workflow with a small context-owned callback operator.
- `integrator_rk4.c`: create an RK4 integrator through the registry and advance
  a real field with a drift callback.
- `stochastic_noise.c`: run two seeded stochastic Euler paths and verify they
  remain deterministic.
- `special_math.c`: call stable special functions and finite-ladder helpers.

Keep examples short enough to read in one sitting. Prefer focused public entry
points and document when an example demonstrates an experimental surface.
