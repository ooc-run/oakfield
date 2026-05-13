/**
 * @file operator_semilinear.c
 * @brief Semilinear operator classification for ETDRK4 split planning.
 *
 * The classifier recognizes exact linear flows and dt-scaled additive forcing
 * operators that can participate in semilinear integrator plans. It inspects
 * registered operator metadata and per-operator configs without mutating the
 * simulation context or registered operators.
 */
#include "operator_semilinear.h"

#include "oakfield/sim_context.h"
#include "oakfield/operators/utility/copy.h"
#include "oakfield/operators/diffusion/laplacian.h"
#include "oakfield/operators/utility/scale.h"
#include "oakfield/operators/stimulus/stimulus.h"
#include "oakfield/operators/stimulus/airy_beam.h"
#include "oakfield/operators/stimulus/bessel_beam.h"
#include "oakfield/operators/stimulus/checkerboard.h"
#include "oakfield/operators/stimulus/chladni.h"
#include "oakfield/operators/stimulus/cylindrical_wave_emitter.h"
#include "oakfield/operators/stimulus/gaussian.h"
#include "oakfield/operators/stimulus/hermite_gaussian_beam.h"
#include "oakfield/operators/stimulus/laplace_beltrami.h"
#include "oakfield/operators/stimulus/lissajous.h"
#include "oakfield/operators/stimulus/log_polar.h"
#include "oakfield/operators/stimulus/log_spectral_grid.h"
#include "oakfield/operators/stimulus/moire.h"
#include "oakfield/operators/stimulus/morlet_field.h"
#include "oakfield/operators/stimulus/optical_vortex.h"
#include "oakfield/operators/stimulus/posenc.h"
#include "oakfield/operators/stimulus/random_fourier.h"
#include "oakfield/operators/stimulus/sinusoidal.h"
#include "oakfield/operators/stimulus/spectral_lines.h"
#include "oakfield/operators/stimulus/spectral_shells.h"
#include "oakfield/operators/stimulus/steerable_wavelet.h"
#include "oakfield/operators/stimulus/traveling_wave_packet.h"
#include "oakfield/operators/stimulus/wave_modes.h"
#include "oakfield/operators/stimulus/zone_plate.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

SimSemilinearTraits sim_operator_semilinear_traits_defaults(void) {
    SimSemilinearTraits traits = { .role        = SIM_SEMILINEAR_ROLE_UNSUPPORTED,
                                   .clock_mode  = SIM_CLOCK_FROM_TIME_PARAM,
                                   .scale_by_dt = false,
                                   .accumulate  = false,
                                   .input_field = SIZE_MAX,
                                   .output_field = SIZE_MAX };
    return traits;
}

static bool sim_operator_semilinear_exact_linear_name(const char* name) {
    if (name == NULL) {
        return false;
    }

    return strcmp(name, "linear_dissipative") == 0 || strcmp(name, "dispersion") == 0 ||
           strcmp(name, "phase_rotate") == 0 || strcmp(name, "linear_spectral_fusion") == 0;
}

static void sim_operator_semilinear_set_dt_scaled_increment(SimSemilinearTraits* traits,
                                                            SimClockMode         clock_mode,
                                                            bool                 accumulate,
                                                            size_t               input_field,
                                                            size_t               output_field) {
    if (traits == NULL) {
        return;
    }

    *traits            = sim_operator_semilinear_traits_defaults();
    traits->role       = SIM_SEMILINEAR_ROLE_DT_SCALED_INCREMENT;
    traits->clock_mode = clock_mode;
    traits->scale_by_dt = true;
    traits->accumulate = accumulate;
    traits->input_field = input_field;
    traits->output_field = output_field;
}

static bool sim_operator_semilinear_classify_forcing(struct SimContext*   context,
                                                     size_t               operator_index,
                                                     const char*          abstract_id,
                                                     size_t               target_field_index,
                                                     SimSemilinearTraits* out_traits) {
#define SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(id_literal, config_type, config_fn)                      \
    if (strcmp(abstract_id, id_literal) == 0) {                                                        \
        config_type config = { 0 };                                                                    \
        if (config_fn(context, operator_index, &config) != SIM_RESULT_OK || !config.scale_by_dt) {    \
            return false;                                                                              \
        }                                                                                              \
        sim_operator_semilinear_set_dt_scaled_increment(                                               \
            out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);                        \
        return true;                                                                                   \
    }

    if (context == NULL || abstract_id == NULL || out_traits == NULL) {
        return false;
    }

    if (strcmp(abstract_id, "stimulus_sine") == 0 || strcmp(abstract_id, "stimulus_standing") == 0 ||
        strcmp(abstract_id, "stimulus_chirp") == 0) {
        SimStimulusSinusoidalConfig config = { 0 };
        if (sim_stimulus_sinusoidal_config(context, operator_index, &config) == SIM_RESULT_OK) {
            if (config.scale_by_dt && !config.fixed_clock) {
                sim_operator_semilinear_set_dt_scaled_increment(
                    out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
                return true;
            }
            return false;
        }
        if (strcmp(abstract_id, "stimulus_sine") == 0) {
            StimulusOperatorConfig simple = { 0 };
            if (sim_stimulus_config(context, operator_index, &simple) == SIM_RESULT_OK &&
                simple.scale_by_dt) {
                sim_operator_semilinear_set_dt_scaled_increment(
                    out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
                return true;
            }
        }
        return false;
    }

    if (strcmp(abstract_id, "stimulus_gaussian_pulse") == 0) {
        SimStimulusGaussianConfig config = { 0 };
        if (sim_stimulus_gaussian_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || config.fixed_clock) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(
            out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
        return true;
    }

    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_checkerboard", SimStimulusCheckerboardConfig, sim_stimulus_checkerboard_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_moire", SimStimulusMoireConfig, sim_stimulus_moire_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_wave_modes", SimStimulusWaveModesConfig, sim_stimulus_wave_modes_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_lissajous", SimStimulusLissajousConfig, sim_stimulus_lissajous_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_chladni", SimStimulusChladniConfig, sim_stimulus_chladni_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_log_polar", SimStimulusLogPolarConfig, sim_stimulus_log_polar_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_posenc", SimStimulusPosEncConfig, sim_stimulus_posenc_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_zone_plate", SimStimulusZonePlateConfig, sim_stimulus_zone_plate_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_optical_vortex", SimStimulusOpticalVortexConfig, sim_stimulus_optical_vortex_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_airy_beam", SimStimulusAiryBeamConfig, sim_stimulus_airy_beam_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_morlet_field", SimStimulusMorletFieldConfig, sim_stimulus_morlet_field_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_traveling_wave_packet",
                                          SimStimulusTravelingWavePacketConfig,
                                          sim_stimulus_traveling_wave_packet_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE(
        "stimulus_bessel_beam", SimStimulusBesselBeamConfig, sim_stimulus_bessel_beam_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_cylindrical_wave_emitter",
                                          SimStimulusCylindricalWaveEmitterConfig,
                                          sim_stimulus_cylindrical_wave_emitter_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_laplace_beltrami",
                                          SimStimulusLaplaceBeltramiConfig,
                                          sim_stimulus_laplace_beltrami_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_log_spectral_grid",
                                          SimStimulusLogSpectralGridConfig,
                                          sim_stimulus_log_spectral_grid_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_steerable_wavelet",
                                          SimStimulusSteerableWaveletConfig,
                                          sim_stimulus_steerable_wavelet_config);
    SIM_SEMILINEAR_CHECK_STATELESS_SOURCE("stimulus_hermite_gaussian_beam",
                                          SimStimulusHermiteGaussianBeamConfig,
                                          sim_stimulus_hermite_gaussian_beam_config);

    if (strcmp(abstract_id, "stimulus_random_fourier") == 0) {
        SimStimulusRandomFourierConfig config = { 0 };
        if (sim_stimulus_random_fourier_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || config.fixed_clock) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(
            out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
        return true;
    }

    if (strcmp(abstract_id, "stimulus_spectral_lines") == 0) {
        SimStimulusSpectralLinesConfig config = { 0 };
        if (sim_stimulus_spectral_lines_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || config.fixed_clock) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(
            out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
        return true;
    }

    if (strcmp(abstract_id, "stimulus_spectral_shells") == 0) {
        SimStimulusSpectralShellsConfig config = { 0 };
        if (sim_stimulus_spectral_shells_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || config.fixed_clock) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(
            out_traits, SIM_CLOCK_FROM_TIME_PARAM, false, SIZE_MAX, SIZE_MAX);
        return true;
    }

    if (strcmp(abstract_id, "scale") == 0) {
        SimScaleOperatorConfig config = { 0 };
        if (sim_scale_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || !config.accumulate ||
            config.input_field != target_field_index || config.output_field != target_field_index) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(out_traits,
                                                        SIM_CLOCK_FROM_TIME_PARAM,
                                                        true,
                                                        config.input_field,
                                                        config.output_field);
        return true;
    }

    if (strcmp(abstract_id, "copy") == 0) {
        SimCopyOperatorConfig config = { 0 };
        if (sim_copy_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || !config.accumulate ||
            config.input_field != target_field_index || config.output_field != target_field_index) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(out_traits,
                                                        SIM_CLOCK_FROM_TIME_PARAM,
                                                        true,
                                                        config.input_field,
                                                        config.output_field);
        return true;
    }

    if (strcmp(abstract_id, "laplacian") == 0) {
        SimLaplacianOperatorConfig config = { 0 };
        if (sim_laplacian_config(context, operator_index, &config) != SIM_RESULT_OK ||
            !config.scale_by_dt || !config.accumulate ||
            config.input_field != target_field_index || config.output_field != target_field_index) {
            return false;
        }
        sim_operator_semilinear_set_dt_scaled_increment(out_traits,
                                                        SIM_CLOCK_FROM_TIME_PARAM,
                                                        true,
                                                        config.input_field,
                                                        config.output_field);
        return true;
    }

#undef SIM_SEMILINEAR_CHECK_STATELESS_SOURCE

    return false;
}

SimResult sim_operator_classify_semilinear(struct SimContext*   context,
                                           size_t               operator_index,
                                           size_t               target_field_index,
                                           SimSemilinearTraits* out_traits) {
    SimOperator*         op;
    const char*          abstract_id;
    SimSemilinearTraits  traits;

    if (context == NULL || out_traits == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    traits = sim_operator_semilinear_traits_defaults();
    if (op->info.is_noise) {
        *out_traits = traits;
        return SIM_RESULT_OK;
    }

    abstract_id = sim_operator_abstract_id(op);
    if (sim_operator_semilinear_exact_linear_name(abstract_id)) {
        traits.role = SIM_SEMILINEAR_ROLE_EXACT_LINEAR_FLOW;
        *out_traits = traits;
        return SIM_RESULT_OK;
    }

    if (sim_operator_semilinear_classify_forcing(
            context, operator_index, abstract_id, target_field_index, &traits)) {
        *out_traits = traits;
        return SIM_RESULT_OK;
    }

    if (!op->info.is_linear) {
        traits.role = SIM_SEMILINEAR_ROLE_GENERAL_NONLINEAR;
    }

    *out_traits = traits;
    return SIM_RESULT_OK;
}
