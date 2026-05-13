# Contributing To Oakfield

Thanks for helping make the package easier to build, test, and
extend.

## Before You Start

- Read `README.md` for the subsystem map and supported build configuration.
- Prefer focused changes with tests or examples near the subsystem they touch.
- Treat `*_internal.h`, `split_internal.h`, and backend-private headers as
  implementation details.

## Build And Test

From the repository root:

```sh
cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release
cmake --build /build
ctest --test-dir /build --output-on-failure
```

Useful opt-in lanes:

```sh
cmake -S . -B /build-docs -DOAKFIELD_BUILD_DOCS=ON
cmake --build /build-docs --target oakfield_docs

cmake -S . -B /build-bench -DOAKFIELD_BUILD_BENCHMARKS=ON
cmake --build /build-bench --target benchmark_core
```

## Coding Expectations

- Keep public headers focused on stable contracts, ownership, lifetime,
  determinism, thread-safety, and numerical assumptions.
- Move future work into `docs/roadmap.md` or issue tracking rather than adding
  TODO notes to public headers.
- Mark experimental APIs visibly in docs, examples, tests, and benchmarks.
- Keep CPU behavior as the correctness reference backend.
- Add deterministic seeds and documented tolerances for numerical tests.
- Keep generated build trees, benchmark timing output, and generated API docs
  out of source control unless a release process intentionally versions them.

## Issue Types

Use the repository issue templates for bug reports, numerical accuracy reports,
backend support, documentation gaps, and operator requests. Include the
build options and platform whenever behavior depends on compiler,
backend, fast-math, OpenMP, vDSP, CUDA, or Metal support.

## License

Oakfield is released under the GNU Affero General Public License
v3.0 or later. See `LICENSE`.
