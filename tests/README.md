# Tests

This directory contains tests adapted for the standalone package.

The first migration pass reviewed the base test suite and selected CPU-only
coverage that maps cleanly onto `/*` without Lua, CLI, UI, server,
Metal, CUDA, or app-runtime dependencies. The detailed base-test map lives in
[`base-test-inventory.md`](base-test-inventory.md).

## Running

```sh
cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release
cmake --build /build --target test
ctest --test-dir /build --output-on-failure
```

The initial suite is intentionally small and should run quickly. Experimental
q-method smoke tests carry the `experimental` and `math` CTest labels; they
verify basic API behavior without claiming stable accuracy or performance
contracts. Zeta/Xi tests are part of the supported math lane and use the `math`
and `zeta` labels. Migrated tests also carry subsystem labels such as `core`,
`backend`, `public`, `neural`, `stimulus`, `integrator`, `runtime`, and `legacy`
so focused runs can use `ctest -L`.
