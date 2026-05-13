# Oakfield source inventory.
#
# Paths are relative to OAKFIELD_ROOT. Keep these groups aligned with the
# subsystem map in /README.md so new contributors can find the owning
# area for a source file before editing CMake.

set(OAKFIELD_SPECIAL_MATH_SOURCES
    core/src/math/special_defaults.c
    core/src/math/special_common.c
    core/src/math/special_diagnostics.c
    core/src/math/special_mp.c
    core/src/math/special_eval.c
    core/src/math/digamma.c
    core/src/math/trigamma.c
    core/src/math/tetragamma.c
    core/src/math/digamma_square.c
    core/src/math/harmonic_ladder.c
    core/src/math/finite_ladder.c
    core/src/math/q_ladder.c
    core/src/math/qnumber.c
    core/src/math/qzeta.c
    core/src/math/qdigamma.c
)

# Core runtime, field, context, KernelIR, and scheduling support.
set(OAKFIELD_CORE_RUNTIME_SOURCES
    core/src/field.c
    core/src/field_patch.c
    core/src/sim_buffer.c
    core/src/neural_models.c
    core/src/neural_tensor_map.c
    core/src/graph_ir.c
    core/src/kernel_ir.c
    core/src/kernel_ir_builder.c
    core/src/kernel_ir_eval.c
    core/src/kernel_ir_eval_complex.c
    core/src/kernel_ir_eval_domain.c
    core/src/kernel_ir_mathview.c
    core/src/operator_identity.c
    core/src/operator.c
    core/src/operator_semilinear.c
    core/src/sim_context.c
    core/src/sim_world.c
    core/src/sim_seed.c
    core/src/sim_noise_source.c
    core/src/sim_runtime_state.c
    core/src/sim_scheduler_state.c
    core/src/sim_integrator_state.c
    core/src/special_functions.c
    core/src/plane_chart.c
    core/src/operator_split.c
    core/src/async_logger.c
)

# Stable math foundations plus experimental q-methods and special-function
# dispatch used by the public math entry points.
set(OAKFIELD_CORE_MATH_SOURCES
    core/src/math/airy.c
    core/src/math/bessel.c
    core/src/math/fourier.c
    ${OAKFIELD_SPECIAL_MATH_SOURCES}
    core/src/math/theta.c
)

# Operator shared helpers.
set(OAKFIELD_OPERATOR_COMMON_SOURCES
    core/src/operators/common/operator_utils.c
    core/src/operators/common/nd_neighbors.c
    core/src/operators/common/warp_safety.c
    core/src/operators/common/fft_plan.c
)

# Stimulus and source-field operators.
set(OAKFIELD_OPERATOR_STIMULUS_SOURCES
    core/src/operators/stimulus/static_cache.c
    core/src/operators/stimulus/octave_noise_common.c
    core/src/operators/stimulus/stimulus.c
    core/src/operators/stimulus/sinusoidal.c
    core/src/operators/stimulus/gaussian.c
    core/src/operators/stimulus/heat_kernel.c
    core/src/operators/stimulus/laplace_beltrami.c
    core/src/operators/stimulus/zone_plate.c
    core/src/operators/stimulus/airy_beam.c
    core/src/operators/stimulus/hermite_gaussian_beam.c
    core/src/operators/stimulus/traveling_wave_packet.c
    core/src/operators/stimulus/cylindrical_wave_emitter.c
    core/src/operators/stimulus/bessel_beam.c
    core/src/operators/stimulus/chladni.c
    core/src/operators/stimulus/wave_modes.c
    core/src/operators/stimulus/optical_vortex.c
    core/src/operators/stimulus/spectral_lines.c
    core/src/operators/stimulus/random_fourier.c
    core/src/operators/stimulus/spectral_shells.c
    core/src/operators/stimulus/fourier.c
    core/src/operators/stimulus/white_noise.c
    core/src/operators/stimulus/checkerboard.c
    core/src/operators/stimulus/moire.c
    core/src/operators/stimulus/lissajous.c
    core/src/operators/stimulus/posenc.c
    core/src/operators/stimulus/log_polar.c
    core/src/operators/stimulus/zeta_plane_slice.c
    core/src/operators/stimulus/log_spectral_grid.c
    core/src/operators/stimulus/morlet_field.c
    core/src/operators/stimulus/steerable_wavelet.c
    core/src/operators/stimulus/rd_seed.c
    core/src/operators/stimulus/digamma_square.c
    core/src/operators/stimulus/fbm.c
    core/src/operators/stimulus/worley_noise.c
    core/src/operators/stimulus/multifractal.c
    core/src/operators/stimulus/hybrid_fbm.c
    core/src/operators/stimulus/ridged_noise.c
    core/src/operators/stimulus/turbulence.c
    core/src/operators/stimulus/gabor.c
)

# Measurement and observer operators.
set(OAKFIELD_OPERATOR_MEASUREMENT_SOURCES
    core/src/operators/measurement/sound_observation.c
    core/src/operators/measurement/phase_feature.c
    core/src/operators/measurement/sieve.c
    core/src/operators/measurement/minimal_convolution.c
    core/src/operators/measurement/sdr_observer.c
)

# Diffusion, spectral, and dissipative operators.
set(OAKFIELD_OPERATOR_DIFFUSION_SOURCES
    core/src/operators/diffusion/linear_dissipative.c
    core/src/operators/diffusion/linear_spectral_fusion.c
    core/src/operators/diffusion/fractional_memory.c
    core/src/operators/diffusion/laplacian.c
    core/src/operators/diffusion/dispersion.c
)

# Noise and stochastic-process operators.
set(OAKFIELD_OPERATOR_NOISE_SOURCES
    core/src/operators/noise/ornstein_uhlenbeck.c
    core/src/operators/noise/stochastic_noise.c
)

# Advection, differential, and coordinate-derivative operators.
set(OAKFIELD_OPERATOR_ADVECTION_SOURCES
    core/src/operators/advection/analytic_warp.c
    core/src/operators/advection/gradient.c
    core/src/operators/advection/divergence.c
    core/src/operators/advection/curl.c
    core/src/operators/advection/spatial_derivative.c
)

# Coupling and mixing operators.
set(OAKFIELD_OPERATOR_COUPLING_SOURCES
    core/src/operators/coupling/segmented_sieve_mark.c
    core/src/operators/coupling/segmented_sieve_mark_batch.c
    core/src/operators/coupling/mask_apply.c
    core/src/operators/coupling/mixer.c
    core/src/operators/coupling/metal_mix.c
)

# Utility operators for field movement, scaling, and representation changes.
set(OAKFIELD_OPERATOR_UTILITY_SOURCES
    core/src/operators/utility/phase_rotate.c
    core/src/operators/utility/coordinate.c
    core/src/operators/utility/copy.c
    core/src/operators/utility/scale.c
    core/src/operators/utility/fft_convert.c
    core/src/operators/utility/zero_field.c
)

# Reaction and nonlinear operators.
set(OAKFIELD_OPERATOR_REACTION_SOURCES
    core/src/operators/reaction/remainder.c
)

set(OAKFIELD_OPERATOR_NONLINEAR_SOURCES
    core/src/operators/nonlinear/math_operator.c
    core/src/operators/nonlinear/complex_math.c
    core/src/operators/nonlinear/chaos_map.c
    core/src/operators/nonlinear/hysteretic.c
)

# Neural and learned-operator integration points.
set(OAKFIELD_OPERATOR_NEURAL_SOURCES
    core/src/operators/neural/neural_infer.c
    core/src/operators/neural/neural_hybrid.c
)

# Runtime control operators.
set(OAKFIELD_OPERATOR_THERMOSTAT_SOURCES
    core/src/operators/thermostat/thermostat.c
)

# Field analysis, topology, and lightweight profiling helpers.
set(OAKFIELD_FIELD_ANALYSIS_SOURCES
    core/src/field/sim_field_stats.c
    core/src/field/sim_field_stats_runtime.c
    core/src/field/sim_field_topology.c
    core/src/field/sim_field_topology_runtime.c
    core/src/field/sim_profiler.c
    core/src/field/sim_flux_lens.c
)

# Backend implementations. CPU is always built as the correctness reference;
# CUDA and Metal are experimental optional backends gated by top-level CMake
# options and platform/toolchain checks.
set(OAKFIELD_BACKEND_CPU_SOURCES
    backends/src/cpu/backend_cpu.c
)

set(OAKFIELD_BACKEND_CUDA_SOURCES
    backends/src/cuda/backend_cuda.cu
)

set(OAKFIELD_BACKEND_METAL_SOURCES
    backends/src/metal/backend_metal.mm
    backends/src/metal/backend_metal_debug.mm
)

# Aggregated simcore library sources.
set(OAKFIELD_SIMCORE_BASE_SOURCES
    ${OAKFIELD_CORE_RUNTIME_SOURCES}
    ${OAKFIELD_CORE_MATH_SOURCES}
    ${OAKFIELD_OPERATOR_COMMON_SOURCES}
    ${OAKFIELD_OPERATOR_STIMULUS_SOURCES}
    ${OAKFIELD_OPERATOR_MEASUREMENT_SOURCES}
    ${OAKFIELD_OPERATOR_DIFFUSION_SOURCES}
    ${OAKFIELD_OPERATOR_NOISE_SOURCES}
    ${OAKFIELD_OPERATOR_ADVECTION_SOURCES}
    ${OAKFIELD_OPERATOR_COUPLING_SOURCES}
    ${OAKFIELD_OPERATOR_UTILITY_SOURCES}
    ${OAKFIELD_OPERATOR_REACTION_SOURCES}
    ${OAKFIELD_OPERATOR_NONLINEAR_SOURCES}
    ${OAKFIELD_OPERATOR_NEURAL_SOURCES}
    ${OAKFIELD_OPERATOR_THERMOSTAT_SOURCES}
    ${OAKFIELD_FIELD_ANALYSIS_SOURCES}
    ${OAKFIELD_BACKEND_CPU_SOURCES}
)

# Zeta/Xi sources gated by OAKFIELD_ENABLE_ZETA_CORE.
set(OAKFIELD_SIMCORE_ZETA_SOURCES
    core/src/math/loggamma.c
    core/src/math/zeta.c
    core/src/math/xi.c
)

# Integrator schemes and registry.
set(OAKFIELD_SIMINTEGRATOR_SOURCES
    integrators/src/backward_euler.c
    integrators/src/crank_nicolson.c
    integrators/src/etdrk4.c
    integrators/src/euler.c
    integrators/src/heun.c
    integrators/src/integrator.c
    integrators/src/integrator_registry.c
    integrators/src/rk4.c
    integrators/src/rkf45.c
    integrators/src/subordination.c
)
