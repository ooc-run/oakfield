# Oakfield

Oakfield is the open-source numerical engine behind the Oakfield Operator Calculus
family of applications.

It provides C APIs for multidimensional fields, operator registration,
KernelIR evaluation, time integrators, special math helpers, and CPU reference
execution.

## Links

- API Documentation: [ooc-run.github.io/oakfield](https://ooc-run.github.io/oakfield)
- Company Website: [ooc.run](https://ooc.run)

## Status

- License: GNU Affero General Public License v3.0 or later; see `LICENSE`.
- Reference backend: CPU.
- Optional backend work: CUDA and Metal are experimental
- Supported math surface: classical digamma/trigamma/tetragamma, finite ladder
  helpers, Airy, Bessel, Fourier, theta, Riemann zeta, completed xi, and
  related special helpers, subject to the documented input domains in their
  headers.
- 60+ supported operators: advection, diffusion, coupling, noise, measurement, neural, stimulus
- Experimental math surface: q-zeta, q-digamma, q-number, and q-ladder methods.
  Their APIs, error models, convergence behavior, branch selection, and
  performance may change.

## Coming Soon

- Lua API
- Runtime

## Build

System requirements for the default CPU build:

- CMake 3.20 or newer.
- A C99 compiler and a C++17 compiler.
- POSIX threads support. CMake resolves this through `Threads::Threads`.
- A standard build tool supported by CMake, such as Make, Ninja, or Xcode.

macOS requirements:

- Xcode Command Line Tools or Xcode with Apple Clang. Install the command-line
  tools with `xcode-select --install`.
- CMake from Homebrew, MacPorts, Kitware, or another trusted package source.
- The CMake build currently sets `CMAKE_OSX_DEPLOYMENT_TARGET` to `15.7`.
- Accelerate/vDSP is auto-detected on Apple platforms when
  `OAKFIELD_ENABLE_VDSP=ON`. No separate package is needed.
- The experimental Metal backend requires Apple platform SDKs with Metal and
  Foundation, and must be enabled explicitly with `-DOAKFIELD_ENABLE_METAL=ON`.

Linux requirements:

- GCC or Clang with C99 and C++17 support.
- GNU Make or Ninja, plus CMake 3.20 or newer. On Debian/Ubuntu systems, the
  default build dependencies are typically available with
  `sudo apt install build-essential cmake ninja-build`.
- The CPU backend can use OpenMP when CMake finds compiler/runtime support.
  Without OpenMP, the default build still completes with CPU paths enabled.
- The experimental CUDA backend requires a CUDA Toolkit installation that
  provides the CUDA driver library and NVRTC, and must be enabled explicitly
  with `-DOAKFIELD_ENABLE_CUDA=ON`.

Optional developer tools:

- Doxygen for `OAKFIELD_BUILD_DOCS=ON`.
- `pkg-config`, MPFR, and GMP for `OAKFIELD_BUILD_APPROX_TOOLS=ON`.
- GSL, Boost.Math headers, MPFR, and GMP for additional benchmark comparison
  rows when `OAKFIELD_BUILD_BENCHMARKS=ON`.
- `clang-format` for the optional public-header formatting check in the test
  suite.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Useful build options:

- `OAKFIELD_BUILD_TESTS`: builds CTest-based tests. Enabled by default.
- `OAKFIELD_BUILD_EXAMPLES`: builds small C example programs. Enabled by default.
- `OAKFIELD_BUILD_DOCS`: adds the optional `oakfield_docs` Doxygen target. Disabled by default.
- `OAKFIELD_BUILD_BENCHMARKS`: builds opt-in developer benchmark executables. Disabled by default.
- `OAKFIELD_ENABLE_EXTENDED_FEATURES`: enables the extended feature
  set used by the current operator catalogue. Enabled by default.
- `OAKFIELD_ENABLE_FAST_MATH`: enables aggressive floating-point optimizations for
  most engine code. Some special-math sources are built without fast-math.
- `OAKFIELD_ENABLE_SYMBOLIC_KERNELS`: enables symbolic KernelIR paths where supported.
- `OAKFIELD_ENABLE_DIAGNOSTICS`: enables diagnostic code paths.
- `OAKFIELD_ENABLE_ZETA_CORE`: includes Zeta/Xi sources.
- `OAKFIELD_ENABLE_OPENMP`: enables OpenMP for CPU paths when available.
- `OAKFIELD_ENABLE_VDSP`: enables Accelerate/vDSP on Apple platforms when available.
- `OAKFIELD_ENABLE_CUDA`: builds the experimental CUDA backend when CMake can find a
  CUDA Toolkit with the driver and NVRTC libraries. Disabled by default.
- `OAKFIELD_ENABLE_METAL`: builds the experimental Metal backend on Apple platforms.
  Disabled by default.

Generated build directories such as `build`, `build-ci`, and other `build-*`
paths are ignored local artifacts and should not be included in source releases.

## Install And Downstream Use

The CMake project installs the public include tree, `simcore`,
`simintegrators`, and CMake package metadata for config-mode discovery:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /tmp/oakfield
```

Downstream CMake projects can then consume the installed package without
referencing source-tree private paths:

```cmake
find_package(oakfield CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE oakfield::simcore)
```

Link `oakfield::simintegrators` as well when creating integrators
directly. Public headers are installed under `include/oakfield`.

## Tests

The first test slice lives under `tests` and covers portable CPU-only
field, buffer, patch, scalar-domain, backend support-level, public-header,
stable special-math smoke paths, supported Zeta/Xi smoke and regression paths,
and labeled experimental q-method smoke paths adapted from the base test suite.
Tests are enabled by default:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Most smoke tests are quick; slow-labeled Zeta/Xi visualization checks can take
several seconds on a normal developer machine. Failures usually indicate a core
ABI/layout regression, a special-function accuracy regression, or a build
configuration mismatch.

Experimental q-method tests carry the `experimental` CTest label and do not
represent stable accuracy or performance promises.

## Examples

Small C examples live under `examples` and mirror the public workflows used by
the Lua and runtime layers: fields, context-owned operators, CPU KernelIR
launch, integrators, seeded stochastic stepping, and stable math calls. Examples
include public headers rather than private source-tree paths.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOAKFIELD_BUILD_EXAMPLES=ON
cmake --build build --target example_minimal_field example_operator_context example_integrator_rk4
build/bin/example_minimal_field
```

## Benchmarks

Benchmarks are opt-in developer tools and are not registered as CTest
correctness checks. Configure a dedicated benchmark build with optional docs,
tests, and examples disabled when you only want the benchmark executables:

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
```

Run the benchmark executables from the build tree:

```sh
build-bench/bin/benchmark_core --elements=65536 --iterations=128
build-bench/bin/benchmark_experimental_math --iterations=32
build-bench/bin/benchmark_special_functions --iterations=128 --samples=256
build-bench/bin/benchmark_diffusion --iterations=16
build-bench/bin/benchmark_integrators --iterations=128 --shape=1024
build-bench/bin/benchmark_segmented_sieve 200000 32768
```

`benchmark_special_functions` auto-detects optional GSL, Boost.Math,
and MPFR rows at configure time. Boost.Math detection checks compiler defaults
plus common Homebrew include and Cellar paths; if Boost lives somewhere else,
add `-DOAKFIELD_BENCH_BOOST_INCLUDE_DIR=/path/to/include` or point directly at
the `boost/` directory.

To include experimental GPU backend code in the same build, add
`-DOAKFIELD_ENABLE_CUDA=ON` on hosts with the CUDA Toolkit installed, and add
`-DOAKFIELD_ENABLE_METAL=ON` on Apple platforms with the Metal framework available.
CUDA and Metal can be enabled together only on machines that provide both
toolchains.

Use benchmark output for local before/after comparisons on the same machine,
problem size, iteration count, compiler, and build configuration. Generated
timing results should stay out of source releases unless a release process
intentionally versions them.

## Approximation Tools

Special-function coefficient work lives behind an opt-in developer build:

```sh
cmake -S . -B build-approx \
  -DOAKFIELD_BUILD_TESTS=OFF \
  -DOAKFIELD_BUILD_EXAMPLES=OFF \
  -DOAKFIELD_BUILD_APPROX_TOOLS=ON
cmake --build build-approx --target oakfield_minimax
```

`build-approx/bin/oakfield_minimax` generates MPFR-backed Chebyshev-basis
polynomial and rational minimax candidates for functions such as digamma,
Airy Ai, and Bessel Jn. Treat emitted coefficients as candidates: validate
them against MPFR tests and local benchmarks before moving them into runtime
math code.

## API Docs

Public headers contain Doxygen-style docblocks, grouped by module for fields,
buffers, contexts, operators, KernelIR, integrators, backends, special
functions, Zeta/Xi APIs, and experimental q-method APIs. The build
includes an optional Doxygen target that writes generated output to the build
tree. The default docs target is focused on public headers and Markdown notes;
private implementation docs can be added later as a separate developer-docs
mode.

To generate docs when Doxygen is installed:

```sh
cmake -S . -B build-docs -DOAKFIELD_BUILD_DOCS=ON
cmake --build build-docs --target oakfield_docs
```

## Directory Map

- `core/src`: core field, buffer, context, operator, KernelIR, runtime, and math
  implementation.
- `core/src/operators`: split-operator catalogue grouped by domain, such as
  stimulus, diffusion, advection, measurement, coupling, utility, noise,
  nonlinear, neural, reaction, and thermostat.
- `integrators/src`: timestepper APIs, built-in integrator implementations, and
  the runtime registry.
- `backends/src`: backend abstraction plus CPU, CUDA, and Metal implementations.
- `cmake`: source inventory used by the CMake build.
- `docs`: contributor-facing design notes and generated-doc scaffolding.
- `examples`: small buildable programs for fields, contexts, operators,
  integrators, backends, and stable math calls.
- `tools/approx`: opt-in MPFR-backed approximation-generation tooling for
  special-function coefficient work.
- `include/oakfield`: public include tree for downstream users.
  `core/src`, `integrators/src`, and `backends/src` are private build inputs.

## Libraries And Public Surfaces

Oakfield installs two static libraries and a public include tree. The
operator catalogue, CPU backend, special math, field/runtime code, and KernelIR
all live in `simcore`; timestepper implementations live in `simintegrators`.

| Target / header | Contains | Notes |
| --- | --- | --- |
| `oakfield::simcore` | Fields, buffers, contexts, operator registry, split operators, KernelIR, GraphIR support, runtime state, diagnostics, special math, all operators, and the CPU backend. | Link this for most applications. CUDA and Metal sources are compiled into this target only when their CMake options are enabled. |
| `oakfield::simintegrators` | `Integrator`, `IntegratorConfig`, built-in timesteppers, and the integrator registry. | Links against `simcore`. Built-ins include Euler, Heun, RK4, RKF45, backward Euler, Crank-Nicolson, ETDRK4, and subordination. |
| `oakfield/core.h` | Aggregate header for core fields, contexts, operators, KernelIR, runtime state, and diagnostics. | Good starting include for engine-level code. Include `oakfield/backend.h` separately when launching kernels directly. |
| `oakfield/operators/basic.h` | Narrow operator include for common Laplacian, noise, copy, phase-rotate, scale, and zero-field workflows. | Prefer this when examples or downstream users only need the basic set. |
| `oakfield/operators/operators.h` | Aggregate header for the full operator catalogue. | Pulls in advection, coupling, diffusion, measurement, neural, noise, nonlinear, reaction, stimulus, thermostat, and utility operators. |
| `oakfield/math.h` | Special-function entry points, including classical helpers, Zeta/Xi, and experimental q-methods. | Classical special functions plus Zeta/Xi are covered by tests and benchmarks; q-method APIs remain experimental. |

## Core Primitives

The engine is built around a small set of C structs with explicit ownership and
runtime contracts.

| Primitive | Public headers | Role |
| --- | --- | --- |
| `SimField`, `SimFieldLayout`, `SimScalarDomain` | `oakfield/field.h` | Own or wrap multidimensional contiguous storage with shape, strides, representation domain, scalar algebra domain, and allocator metadata. |
| `SimBuffer`, `SimBufferView` | `oakfield/sim_buffer.h` | Typed scalar allocations for auxiliary data that does not need full field layout metadata. |
| `SimContext` | `oakfield/sim_context.h` | Owns fields, registered operators, scheduler state, runtime counters, diagnostics, integrator bindings, memory limits, and deterministic seed state. |
| `SimOperator`, `SimOperatorDescriptor`, `SimOperatorInfo` | `oakfield/operator.h` | Declare executable work, read/write fields, dependencies, category metadata, algebraic flags, determinism, representation hints, and optional KernelIR-backed execution. |
| `SimSplitDescriptor`, `SimSplitSubstep`, `SimSplitPort` | `oakfield/operator_split.h` | Expand a higher-level operator into ordered substeps with shared state, field access declarations, scratch requirements, and dependency edges. |
| `SimIRBuilder`, `SimIRNode`, `SimIRType` | `oakfield/kernel_ir.h` | Build typed expression graphs for backend-independent numeric kernels. Builders own node storage and constant pools. |
| `KernelIR`, `SimKernelIRBinding`, `SimKernelIROutput` | `oakfield/backend.h` | Package a builder with runtime field bindings, output expression roots, optional parameters, required backend features, and complex-lane semantics for launch. |
| `SimBackend` | `oakfield/backend.h` | Initializes and launches CPU, CUDA, or Metal execution engines through a common feature-bit contract. CPU is the correctness reference. |
| `Integrator`, `IntegratorConfig`, `IntegratorRegistry` | `oakfield/integrator.h`, `oakfield/integrator_registry.h` | Advance fields from drift/noise callbacks, manage adaptive timestep diagnostics, and look up built-in or custom schemes. |
| `SimFieldPatch`, `SimPlaneChart`, field stats/topology helpers | `oakfield/field_patch.h`, `oakfield/plane_chart.h`, `oakfield/sim_field_stats*.h`, `oakfield/sim_field_topology*.h` | Support patch iteration, coordinate projections, runtime statistics, topology summaries, and visualization-oriented analysis. |

## KernelIR System

KernelIR is the pointwise expression layer used by operators and backends to
describe numeric work without committing to a concrete execution engine. It is
not a full scheduler by itself; a `KernelIR` package describes one backend
launch, while `SimContext` and the operator scheduler decide when that launch
runs.

The core data path is:

1. A caller initializes a `SimIRBuilder`.
2. Builder helpers append typed `SimIRNode` records and return `SimIRNodeId`
   handles.
3. One or more root expressions are mapped to destination field identifiers via
   `SimKernelIROutput`.
4. `SimKernelIRBinding` records connect KernelIR field identifiers to runtime
   `SimField` storage.
5. A `KernelIR` launch package borrows the builder, bindings, outputs, optional
   parameter array, feature mask, and complex-lane contract for the duration of
   `backend_launch()`.

Builders own expression storage. Kernel launch packages borrow that storage.
This split keeps graph construction reusable while making backend launches cheap
to assemble from live context fields. Operators that expose KernelIR usually
store an owned `SimOperatorKernel` wrapper that clones binding/output maps and
keeps mutable runtime parameter storage.

### KernelIR Expression Primitives

| Primitive group | Node types / helpers | Notes |
| --- | --- | --- |
| Literals and constants | `SIM_IR_NODE_CONSTANT`, `sim_ir_builder_constant*`, `sim_ir_builder_constant_vector*`, `sim_ir_builder_constant_complex` | Scalars, exact signed/unsigned integers, small inline vectors, pooled vector constants, and complex constants. |
| Field, coordinate, and index reads | `SIM_IR_NODE_FIELD_REF`, `SIM_IR_NODE_COORD`, `SIM_IR_NODE_INDEX` | Field references are logical IDs resolved by launch bindings; coordinate/index nodes derive element position from bound field layout. |
| Arithmetic | `ADD`, `SUB`, `MUL`, `DIV`, `POW`, `sim_ir_builder_binary`, `sim_ir_builder_pow` | Typed scalar/vector arithmetic. Real, complex, integer, and modular legality is controlled by `SimScalarDomain`. |
| Differential terms | `SIM_IR_NODE_DIFF`, `sim_ir_builder_diff`, `sim_ir_builder_diff_spec` | Carries operand, axis, spacing, scale, derivative order, stencil order, method, consistency constant, and boundary policy. |
| Noise terms | `SIM_IR_NODE_NOISE`, `sim_ir_builder_noise`, `sim_ir_builder_noise_spec` | Seeded stochastic nodes with amplitude, variance, Ito or Stratonovich law, and uniform or Gaussian distribution metadata. |
| Analytic warp | `SIM_IR_NODE_WARP`, `sim_ir_builder_warp`, `sim_ir_builder_warp_spec` | Digamma/trigamma-family warp profiles with bias, symmetric delta, lambda scale, tolerance, warp level, and continuity guards. |
| Runtime parameters | `SIM_IR_NODE_PARAM`, `sim_ir_builder_param` | Reads `dt`, `step_index`, `sqrt_dt`, or simulation `time` from backend/evaluator callbacks. |
| Complex helpers | `SIM_IR_NODE_COMPLEX_PACK`, `SIM_IR_NODE_COMPLEX_ROTATE` | Pack real/imag lanes and rotate complex values by a scalar phase. Launches declare true-complex or componentwise lane semantics. |
| Unary and discrete helpers | `SIM_IR_NODE_CALL`, `SIM_IR_NODE_FLOOR`, `SIM_IR_NODE_MOD` | Built-in calls include `sin`, `cos`, `exp`, `abs`, `log`, `tanh`, `sinh`, and `sign`, plus floor and modulo. |
| Stateful callback | `SIM_IR_NODE_STATEFUL`, `sim_ir_builder_stateful*` | CPU-oriented callback node for exploratory or stateful expressions; strict deterministic paths should avoid it unless the operator documents replay behavior. |

### Types, Metadata, And Evaluation

Every node carries a `SimIRType`, semantic opcode, inferred shape, warp
classification, and locality flag. `SimIRType` separates scalar versus vector
values from the scalar algebra domain, so the same expression machinery can
reason about real floating-point, complex floating-point, exact integer, and
modular values. Operators can tag reachable nodes with `SimIROpcode` values
such as `diffuse`, `warp`, `noise`, or `flow` so downstream tooling can classify
generated kernels.

The CPU backend is the reference evaluator. It walks each output expression for
each destination element with memoized recursive node evaluation, resolves
fields through launch bindings, evaluates finite differences from bound field
layout, samples seeded noise, and dispatches real, complex, vector-lane, and
scalar-domain-aware paths as needed. Optional CUDA and Metal backends consume
the same `KernelIR` package when compiled in, but may reject packages whose
feature bits or complex semantics are not supported.

`kernel_ir_mathview` renders KernelIR expressions as canonical strings, JSON,
LaTeX outlines, and hashes. This gives tests and external tooling a stable view
of the symbolic expression without depending on backend source generation.

GraphIR sits one level above KernelIR. It composes pointwise KernelIR nodes with
FFT forward/inverse nodes, real-to-complex promotion, cast/copy nodes, and
Hermitian canonicalization. GraphIR validation checks representation and time
contracts, and compatible adjacent pointwise KernelIR nodes can be fused before
execution.

## Available Operators

Operators are registered into a `SimContext` with `sim_add_*` helpers, then
optionally inspected with `sim_*_config(...)` and updated with
`sim_*_update(...)`. The aggregate `oakfield/operators/operators.h` exposes the
full catalogue. The table tracks public registration surfaces backed by
`core/src/operators`; some rows share the same implementation file because a
single source exposes several variants or compatibility wrappers.

| Family | Operator / helper | Source | Purpose |
| --- | --- | --- | --- |
| Advection | `sim_add_analytic_warp_operator` | `advection/analytic_warp.c` | Smooth nonlinear deformation acting on a single field. |
| Advection | `sim_add_curl_operator` | `advection/curl.c` | Finite-difference curl-like scalar from two vector components. |
| Advection | `sim_add_divergence_operator` | `advection/divergence.c` | Finite-difference divergence from two vector components. |
| Advection | `sim_add_gradient_operator` | `advection/gradient.c` | Finite-difference gradient writing X/Y derivative fields. |
| Advection | `sim_add_spatial_derivative_operator` | `advection/spatial_derivative.c` | 1D finite-difference spatial derivative. |
| Coupling | `sim_add_mask_operator` | `coupling/mask_apply.c` | Gate a field by a mask, with optional inversion. |
| Coupling | `sim_add_metal_mix_operator` | `coupling/metal_mix.c` | Metal-friendly linear/crossfade mixer subset with split fallback. |
| Coupling | `sim_add_mixer_operator` | `coupling/mixer.c` | Linear, multiply, crossfade, AM/FM/PM, ring-mod, min/max, and feedback mixing. |
| Coupling | `sim_add_segmented_sieve_mark_operator` | `coupling/segmented_sieve_mark.c` | Exact integer segmented-sieve marking. |
| Coupling | `sim_add_segmented_sieve_mark_batch_operator` | `coupling/segmented_sieve_mark_batch.c` | Batched exact integer segmented-sieve marking. |
| Diffusion | `sim_add_dispersion_operator` | `diffusion/dispersion.c` | Spectral k-dependent phase rotation. |
| Diffusion | `sim_add_fractional_memory_operator` | `diffusion/fractional_memory.c` | History-based fractional derivative of order `q`. |
| Diffusion | `sim_add_laplacian_operator` | `diffusion/laplacian.c` | Finite-difference Laplacian for 1D/2D fields. |
| Diffusion | `sim_add_linear_dissipative_operator` | `diffusion/linear_dissipative.c` | Fractional Laplacian spectral damping. |
| Diffusion | `sim_add_linear_spectral_fusion_operator` | `diffusion/linear_spectral_fusion.c` | Fused spectral dissipation, dispersion, and phase. |
| Measurement | `sim_add_minimal_convolution_operator` | `measurement/minimal_convolution.c` | Small odd-length 1D/2D convolution kernels. |
| Measurement | `sim_add_phase_feature_operator` | `measurement/phase_feature.c` | Phase-feature extraction for real and complex fields. |
| Measurement | `sim_add_sdr_observer_operator` | `measurement/sdr_observer.c` | Experimental SDR observation with RTL-SDR and synthetic fallback status. |
| Measurement | `sim_add_sieve_operator` | `measurement/sieve.c` | Discrete filtering with low/high/band-pass and windowed responses. |
| Measurement | `sim_add_sound_observation_operator` | `measurement/sound_observation.c` | Map field measurements to audio-control observations. |
| Neural | `sim_add_neural_hybrid_operator` | `neural/neural_hybrid.c` | Analytic-plus-residual neural hybrid scaffold. |
| Neural | `sim_add_neural_infer_operator` | `neural/neural_infer.c` | Neural inference operator scaffold. |
| Noise | `sim_add_ornstein_uhlenbeck_operator` | `noise/ornstein_uhlenbeck.c` | Ornstein-Uhlenbeck stochastic process update. |
| Noise | `sim_add_stochastic_noise_operator` | `noise/stochastic_noise.c` | Additive seeded noise with spectral shaping and calculus-law metadata. |
| Nonlinear | `sim_add_chaos_map_operator` | `nonlinear/chaos_map.c` | Discrete chaotic maps for real or complex state fields. |
| Nonlinear | `sim_add_complex_math_operator` | `nonlinear/complex_math.c` | Elementwise complex math transforms and component extraction. |
| Nonlinear | `sim_add_hysteretic_operator` | `nonlinear/hysteretic.c` | Schmitt, play/deadband, and Bouc-Wen hysteresis modes. |
| Nonlinear | `sim_add_elementwise_math_operator` | `nonlinear/math_operator.c` | Discrete-friendly elementwise scalar math. |
| Reaction | `sim_add_remainder_operator` | `reaction/remainder.c` | Measure `f(warped) - f(reference)` with optional accumulation. |
| Stimulus | `sim_add_stimulus_operator` | `stimulus/stimulus.c` | Legacy single-mode sinusoidal forcing. |
| Stimulus | `sim_add_stimulus_airy_beam_operator` | `stimulus/airy_beam.c` | Finite-energy separable Airy beam. |
| Stimulus | `sim_add_stimulus_bessel_beam_operator` | `stimulus/bessel_beam.c` | Integer-order cylindrical Bessel beam. |
| Stimulus | `sim_add_stimulus_checkerboard_operator` | `stimulus/checkerboard.c` | Checkerboard or stripe pattern writes. |
| Stimulus | `sim_add_stimulus_chirp_operator` | `stimulus/sinusoidal.c` | Chirped sinusoidal stimulus. |
| Stimulus | `sim_add_stimulus_chladni_operator` | `stimulus/chladni.c` | Chladni nodal-line plate modes. |
| Stimulus | `sim_add_stimulus_cylindrical_wave_emitter_operator` | `stimulus/cylindrical_wave_emitter.c` | Regularized cylindrical wave emitter. |
| Stimulus | `sim_add_stimulus_digamma_square_operator` | `stimulus/digamma_square.c` | Digamma-square waveform stimulus. |
| Stimulus | `sim_add_digamma_square_operator` | `stimulus/digamma_square.c` | Compatibility wrapper for digamma-square stimulus registration. |
| Stimulus | `sim_add_stimulus_fbm_operator` | `stimulus/fbm.c` | Fractional Brownian motion stimulus with Hurst exponent. |
| Stimulus | `sim_add_stimulus_fourier_operator` | `stimulus/fourier.c` | Bandlimited saw, square, and triangle waveform stimulus. |
| Stimulus | `sim_add_stimulus_gabor_operator` | `stimulus/gabor.c` | Gaussian-windowed sinusoidal Gabor kernel. |
| Stimulus | `sim_add_stimulus_gaussian_operator` | `stimulus/gaussian.c` | Traveling Gaussian envelope modulation. |
| Stimulus | `sim_add_stimulus_heat_kernel_operator` | `stimulus/heat_kernel.c` | Diffusive Gaussian heat-kernel broadening. |
| Stimulus | `sim_add_stimulus_hermite_gaussian_beam_operator` | `stimulus/hermite_gaussian_beam.c` | Hermite-Gaussian beam with separable transverse modes. |
| Stimulus | `sim_add_stimulus_hybrid_fbm_operator` | `stimulus/hybrid_fbm.c` | Hybrid fBm with octave-to-octave cascade weighting. |
| Stimulus | `sim_add_stimulus_laplace_beltrami_operator` | `stimulus/laplace_beltrami.c` | Analytic Laplace-Beltrami eigenmodes on simple manifolds. |
| Stimulus | `sim_add_stimulus_lissajous_operator` | `stimulus/lissajous.c` | Lissajous ridge with Gaussian band shaping. |
| Stimulus | `sim_add_stimulus_log_polar_operator` | `stimulus/log_polar.c` | Log-polar interference with optional spiral phase drift. |
| Stimulus | `sim_add_stimulus_log_spectral_grid_operator` | `stimulus/log_spectral_grid.c` | Log-frequency spectral grid with coordinate controls. |
| Stimulus | `sim_add_stimulus_moire_operator` | `stimulus/moire.c` | Moire interference in 1D/2D coordinate systems. |
| Stimulus | `sim_add_stimulus_morlet_field_operator` | `stimulus/morlet_field.c` | Multi-scale Morlet wavelet field stimulus. |
| Stimulus | `sim_add_stimulus_multifractal_operator` | `stimulus/multifractal.c` | Multiplicative multifractal from centered octave products. |
| Stimulus | `sim_add_stimulus_optical_vortex_operator` | `stimulus/optical_vortex.c` | Optical vortex beam with phase winding. |
| Stimulus | `sim_add_stimulus_posenc_operator` | `stimulus/posenc.c` | NeRF-style positional encoding field. |
| Stimulus | `sim_add_stimulus_random_fourier_operator` | `stimulus/random_fourier.c` | Seeded random Fourier feature fields. |
| Stimulus | `sim_add_stimulus_rd_seed_operator` | `stimulus/rd_seed.c` | Reaction-diffusion seed patterns. |
| Stimulus | `sim_add_stimulus_ridged_noise_operator` | `stimulus/ridged_noise.c` | Ridged fractal noise from a squared ridge profile. |
| Stimulus | `sim_add_stimulus_sine_operator` | `stimulus/sinusoidal.c` | Traveling-wave sinusoidal stimulus. |
| Stimulus | `sim_add_stimulus_spectral_lines_operator` | `stimulus/spectral_lines.c` | Pure frequency spikes and multi-line harmonics. |
| Stimulus | `sim_add_stimulus_spectral_shells_operator` | `stimulus/spectral_shells.c` | Random annular bands in k-space. |
| Stimulus | `sim_add_stimulus_standing_operator` | `stimulus/sinusoidal.c` | Standing-wave sinusoidal stimulus. |
| Stimulus | `sim_add_stimulus_steerable_wavelet_operator` | `stimulus/steerable_wavelet.c` | Simoncelli/Riesz-style steerable wavelet stimulus. |
| Stimulus | `sim_add_stimulus_traveling_wave_packet_operator` | `stimulus/traveling_wave_packet.c` | Gaussian-envelope traveling wave packet. |
| Stimulus | `sim_add_stimulus_turbulence_operator` | `stimulus/turbulence.c` | Turbulence stimulus from centered absolute-value fractal basis. |
| Stimulus | `sim_add_stimulus_wave_modes_operator` | `stimulus/wave_modes.c` | Rectangular wave-equation standing modes. |
| Stimulus | `sim_add_stimulus_white_noise_operator` | `stimulus/white_noise.c` | Seeded Gaussian white-noise injection. |
| Stimulus | `sim_add_stimulus_worley_noise_operator` | `stimulus/worley_noise.c` | Worley/cellular noise with selectable distance metrics. |
| Stimulus | `sim_add_stimulus_zeta_plane_slice_operator` | `stimulus/zeta_plane_slice.c` | Zeta/Xi complex-plane slice visualization. |
| Stimulus | `sim_add_stimulus_zone_plate_operator` | `stimulus/zone_plate.c` | Fresnel-style zone plate with quadratic radial phase. |
| Thermostat | `sim_add_thermostat_operator` | `thermostat/thermostat.c` | Soft energy/lambda regulation. |
| Utility | `sim_add_coordinate_operator` | `utility/coordinate.c` | Coordinate or linear-index field generation. |
| Utility | `sim_add_copy_operator` | `utility/copy.c` | Copy one compatible field into another. |
| Utility | `sim_add_fft_convert` | `utility/fft_convert.c` | Allocate FFT conversion output and transition physical/spectral domains. |
| Utility | `sim_add_phase_rotate_operator` | `utility/phase_rotate.c` | Advance field phase by rate times timestep. |
| Utility | `sim_add_scale_operator` | `utility/scale.c` | Write a scaled copy of one field to another. |
| Utility | `sim_add_zero_field_operator` | `utility/zero_field.c` | Clear a real or complex field in place. |

## Operator Usage Pattern

Most built-in operators follow the same shape: fill a config struct, register
it into a `SimContext`, keep the returned operator index if you want to inspect
or update it later, then execute the context. These snippets assume `ctx` is an
initialized `SimContext`, the named field indices already came from
`sim_context_add_field()`, and production code checks every `SimResult`.

```c
#include <oakfield/core.h>
#include <oakfield/operators/operators.h>

/* Finite-difference Laplacian: field_u -> field_lap. */
size_t lap_op = 0;
SimLaplacianOperatorConfig lap = {
    .input_field = field_u,
    .output_field = field_lap,
    .spacing_x = dx,
    .spacing_y = dy,
    .axis_x = SIM_LAPLACIAN_AXIS_AUTO,
    .axis_y = SIM_LAPLACIAN_AXIS_AUTO,
    .stencil = SIM_LAPLACIAN_STENCIL_CROSS_2,
    .boundary = SIM_IR_BOUNDARY_PERIODIC,
};
SimResult rc = sim_add_laplacian_operator(&ctx, &lap, &lap_op);

/* Accumulate a scaled copy, often used for simple explicit updates. */
size_t scale_op = 0;
SimScaleOperatorConfig scale = {
    .input_field = field_lap,
    .output_field = field_u,
    .scale = 0.01,
    .accumulate = true,
    .scale_by_dt = true,
};
rc = sim_add_scale_operator(&ctx, &scale, &scale_op);

/* Add seeded stochastic forcing to a field. */
size_t noise_op = 0;
StochasticNoiseOperatorConfig noise = {
    .field_index = field_u,
    .sigma = 0.05,
    .tau = 0.2,
    .alpha = 1.0,
    .seed = 1234ULL,
    .law = SIM_IR_NOISE_LAW_ITO,
};
rc = sim_add_stochastic_noise_operator(&ctx, &noise, &noise_op);

/* Write a procedural Gaussian stimulus into a field. */
size_t gaussian_op = 0;
SimStimulusGaussianConfig gaussian = {
    .field_index = field_drive,
    .amplitude = 1.0,
    .sigma_x = 0.15,
    .sigma_y = 0.15,
    .scale_by_dt = false,
};
rc = sim_add_stimulus_gaussian_operator(&ctx, &gaussian, &gaussian_op);

/* Crossfade two fields into an output field. */
size_t mix_op = 0;
SimMixerOperatorConfig mix = {
    .lhs_field = field_a,
    .rhs_field = field_b,
    .output_field = field_mix,
    .lhs_gain = 1.0,
    .rhs_gain = 1.0,
    .mix = 0.35,
    .mode = SIM_MIXER_MODE_CROSSFADE,
};
rc = sim_add_mixer_operator(&ctx, &mix, &mix_op);

/* FFT conversion is a utility registration that allocates the output field. */
size_t spectrum_field = 0;
size_t fft_op = 0;
rc = sim_add_fft_convert(&ctx,
                         field_u,
                         SIM_FFT_CONVERT_FORWARD,
                         false,
                         &spectrum_field,
                         &fft_op);

/* Inspect or update a registered operator by index. */
SimScaleOperatorConfig active_scale = { 0 };
rc = sim_scale_config(&ctx, scale_op, &active_scale);
active_scale.scale = 0.02;
rc = sim_scale_update(&ctx, scale_op, &active_scale);

/* Execute the scheduled operators for one context step. */
sim_context_set_timestep(&ctx, 0.01);
rc = sim_context_execute(&ctx);
if (rc == SIM_RESULT_OK) {
    sim_context_accept_step(&ctx, sim_context_timestep(&ctx));
}
```

## Contributing Notes

- Prefer public headers and examples when learning the API.
- Treat files named `*_internal.h` and private backend headers as implementation
  details.
- Document ownership, lifetime, determinism, numerical assumptions, and
  experimental status when changing public APIs.
- Keep generated files, local build trees, and internal migration notes out of
  release artifacts.
