function(oakfield_add_test name source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LABELS PRIVATE_INCLUDE_DIRS)
    cmake_parse_arguments(OAKFIELD_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${name} ${source})
    oakfield_suppress_duplicate_library_warnings(${name})
    target_link_libraries(${name} PRIVATE simcore)
    if(OAKFIELD_TEST_PRIVATE_INCLUDE_DIRS)
        target_include_directories(${name} PRIVATE ${OAKFIELD_TEST_PRIVATE_INCLUDE_DIRS})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    if(OAKFIELD_TEST_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${OAKFIELD_TEST_LABELS}")
    endif()
endfunction()

function(oakfield_add_operator_test category suffix source)
    oakfield_add_test(operator_${category}_${suffix}
        operators/${category}/${source}
        LABELS operator ${category} ${ARGN})
endfunction()

function(oakfield_add_stimulus_operator_test suffix source)
    oakfield_add_operator_test(stimulus ${suffix} ${source} ${ARGN})
endfunction()

oakfield_add_test(field_buffer_smoke core/test_field_buffer_smoke.c
    LABELS core)
oakfield_add_test(field_patch core/test_field_patch.c
    LABELS core)
oakfield_add_test(neural_tensor_map core/test_neural_tensor_map.c
    LABELS core neural)
oakfield_add_test(plane_chart_stimulus_compat
    core/test_plane_chart_stimulus_compat.c
    LABELS core stimulus)
oakfield_add_test(graph_ir_policy core/test_graph_ir_policy.c
    LABELS core kernel_ir legacy
    PRIVATE_INCLUDE_DIRS "${OAKFIELD_SOURCE_DIR}/core/src")
oakfield_add_test(pole_field core/test_pole_field.c
    LABELS core world)
oakfield_add_test(field_legacy_contracts
    core/test_field_legacy_contracts.c
    LABELS core legacy kernel_ir backend
    PRIVATE_INCLUDE_DIRS "${OAKFIELD_SOURCE_DIR}/core/src")
oakfield_add_test(scalar_domain_legacy_contracts
    core/test_scalar_domain_legacy_contracts.c
    LABELS core scalar legacy)
oakfield_add_stimulus_operator_test(coords_2d test_coords_2d.c)
oakfield_add_stimulus_operator_test(checkerboard_coords test_checkerboard_coords.c)
oakfield_add_stimulus_operator_test(log_spectral_grid_coords test_log_spectral_grid_coords.c)
oakfield_add_stimulus_operator_test(moire_coords test_moire_coords.c)
oakfield_add_stimulus_operator_test(morlet_field_coords test_morlet_field_coords.c)
oakfield_add_stimulus_operator_test(posenc_coords test_posenc_coords.c)
oakfield_add_stimulus_operator_test(rd_seed_coords test_rd_seed_coords.c)
oakfield_add_stimulus_operator_test(spectral_shells_coords test_spectral_shells_coords.c)
oakfield_add_stimulus_operator_test(steerable_wavelet_coords test_steerable_wavelet_coords.c)
oakfield_add_stimulus_operator_test(wavevector_modes test_wavevector_modes.c)
oakfield_add_stimulus_operator_test(airy_beam_complex test_airy_beam_complex.c)
oakfield_add_stimulus_operator_test(bessel_beam_complex test_bessel_beam_complex.c)
oakfield_add_stimulus_operator_test(chladni_complex test_chladni_complex.c)
oakfield_add_stimulus_operator_test(cylindrical_wave_emitter_complex
    test_cylindrical_wave_emitter_complex.c)
oakfield_add_stimulus_operator_test(digamma_square_complex_contracts
    test_digamma_square_complex_contracts.c)
oakfield_add_stimulus_operator_test(fbm_complex test_fbm_complex.c)
oakfield_add_stimulus_operator_test(gabor_complex test_gabor_complex.c)
oakfield_add_stimulus_operator_test(gaussian_complex test_gaussian_complex.c)
oakfield_add_stimulus_operator_test(heat_kernel_complex test_heat_kernel_complex.c)
oakfield_add_stimulus_operator_test(hermite_gaussian_beam_complex
    test_hermite_gaussian_beam_complex.c)
oakfield_add_stimulus_operator_test(hybrid_fbm_complex test_hybrid_fbm_complex.c)
oakfield_add_stimulus_operator_test(laplace_beltrami_complex test_laplace_beltrami_complex.c)
oakfield_add_stimulus_operator_test(lissajous_complex test_lissajous_complex.c)
oakfield_add_stimulus_operator_test(log_polar_complex test_log_polar_complex.c)
oakfield_add_stimulus_operator_test(multifractal_complex test_multifractal_complex.c)
oakfield_add_stimulus_operator_test(optical_vortex_complex test_optical_vortex_complex.c)
oakfield_add_stimulus_operator_test(ridged_noise_complex test_ridged_noise_complex.c)
oakfield_add_stimulus_operator_test(sinusoidal_complex test_sinusoidal_complex.c)
oakfield_add_stimulus_operator_test(traveling_wave_packet_complex
    test_traveling_wave_packet_complex.c)
oakfield_add_stimulus_operator_test(turbulence_complex test_turbulence_complex.c)
oakfield_add_stimulus_operator_test(wave_modes_complex test_wave_modes_complex.c)
oakfield_add_stimulus_operator_test(worley_noise_complex test_worley_noise_complex.c)
oakfield_add_stimulus_operator_test(zone_plate_complex test_zone_plate_complex.c)
oakfield_add_stimulus_operator_test(dt_scaling test_dt_scaling.c timestep)
oakfield_add_stimulus_operator_test(white_noise_dt_scaling test_white_noise_dt_scaling.c
    timestep)
oakfield_add_operator_test(advection analytic_warp test_analytic_warp.c)
oakfield_add_operator_test(advection analytic_warp_complex test_analytic_warp_complex.c)
oakfield_add_operator_test(advection analytic_warp_complex_polar
    test_analytic_warp_complex_polar.c)
oakfield_add_operator_test(advection curl_complex test_curl_complex.c)
oakfield_add_operator_test(advection divergence_complex test_divergence_complex.c)
oakfield_add_operator_test(advection spatial_derivative test_spatial_derivative.c)
oakfield_add_operator_test(diffusion dispersion test_dispersion.c)
oakfield_add_operator_test(diffusion fractional_memory_complex
    test_fractional_memory_complex.c)
oakfield_add_operator_test(diffusion linear_dissipative_fft
    test_linear_dissipative_fft.c)
oakfield_add_operator_test(diffusion linear_spectral_fusion
    test_linear_spectral_fusion.c)
oakfield_add_operator_test(coupling mixer_complex test_mixer_complex.c)
oakfield_add_operator_test(nonlinear hysteretic_complex test_hysteretic_complex.c)
oakfield_add_operator_test(noise ornstein_uhlenbeck_complex
    test_ornstein_uhlenbeck_complex.c)
oakfield_add_operator_test(noise ornstein_uhlenbeck_real
    test_ornstein_uhlenbeck_real.c)
oakfield_add_operator_test(noise stochastic_noise_complex test_stochastic_noise_complex.c)
oakfield_add_operator_test(reaction remainder_complex test_remainder_complex.c)
oakfield_add_operator_test(reaction remainder_complex_polar
    test_remainder_complex_polar.c)
oakfield_add_operator_test(reaction remainder_real_to_complex
    test_remainder_real_to_complex.c)
oakfield_add_operator_test(measurement sieve_complex test_sieve_complex.c)
oakfield_add_operator_test(measurement sdr_observer_operator test_sdr_observer_operator.c
    sdr)
oakfield_add_operator_test(measurement sdr_observer_status test_sdr_observer_status.c
    sdr)
oakfield_add_operator_test(discrete discrete_ops test_discrete_ops.c)
oakfield_add_operator_test(utility fft_convert_properties test_fft_convert_properties.c
    fft)
oakfield_add_test(operator_nonstimulus_dt_scaling
    operators/test_nonstimulus_dt_scaling.c
    LABELS operator timestep)
oakfield_add_test(operator_real_value_support
    operators/test_real_value_support.c
    LABELS operator legacy)
oakfield_add_test(backend_support_levels backends/test_backend_support_levels.c
    LABELS backend)
oakfield_add_test(backend_cpu_vdsp backends/test_backend_cpu_vdsp.c
    LABELS backend platform)
oakfield_add_test(backend_metal_mix backends/test_metal_mix.c
    LABELS backend platform metal)
oakfield_add_test(public_headers public/test_public_headers.c
    LABELS public)
add_test(
    NAME public_header_audit
    COMMAND "${CMAKE_COMMAND}"
            "-DOAKFIELD_PUBLIC_INCLUDE_DIR=${OAKFIELD_PUBLIC_INCLUDE_DIR}"
            -P "${CMAKE_CURRENT_LIST_DIR}/../cmake/PublicHeaderAudit.cmake")
oakfield_add_test(special_math_smoke math/test_special_math_smoke.c
    LABELS math)
oakfield_add_test(special_fallback_hooks math/test_special_fallback_hooks.c
    LABELS math core)
oakfield_add_test(experimental_q_methods_smoke
    math/test_experimental_q_methods_smoke.c
    LABELS experimental math)
oakfield_add_test(integrator_continuous_time_dt_refinement
    integrators/test_continuous_time_dt_refinement.c
    LABELS integrator timestep)
oakfield_add_test(integrator_etdrk4 integrators/test_etdrk4.c
    LABELS integrator)
oakfield_add_test(integrator_regression
    integrators/test_integrator_regression.c
    LABELS integrator)
oakfield_add_test(integrator_subordination_dt_scaling
    integrators/test_subordination_dt_scaling.c
    LABELS integrator timestep)
oakfield_add_test(runtime_step_metrics runtime/test_step_metrics.c
    LABELS runtime)
oakfield_add_test(runtime_world_continuity runtime/test_world_continuity.c
    LABELS runtime world)

if(OAKFIELD_ENABLE_ZETA_CORE)
    oakfield_add_stimulus_operator_test(zeta_plane_slice test_zeta_plane_slice.c
        math zeta slow)
    oakfield_add_test(zeta_smoke
        math/test_zeta_smoke.c
        LABELS math zeta)
    oakfield_add_test(riemann_phase1
        math/test_riemann_phase1.c
        LABELS math zeta)
endif()
