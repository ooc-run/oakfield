/**
 * @file sim_flux_lens.h
 * @brief Flux lens state and skew-Burgers flux utilities.
 */
#ifndef OAKFIELD_SIM_FLUX_LENS_H
#define OAKFIELD_SIM_FLUX_LENS_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>

#include "field.h"
#include "operators/common/fft_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;
struct SimField;

/**
 * @brief Spectral flux landmarks computed from the latest flux-lens analysis.
 */
typedef struct FluxMarks {
    bool valid;            /**< True when the remaining landmark fields are current. */
    double kc;             /**< Critical wavenumber selected from the flux spectrum. */
    double k50;            /**< Wavenumber containing 50 percent of absolute flux work. */
    double k90;            /**< Wavenumber containing 90 percent of absolute flux work. */
    double pi_at_kc;       /**< Flux Pi value sampled at @ref kc. */
    double total_abs_work; /**< Integral/sum of absolute spectral work. */
    double pi_min;         /**< Minimum observed Pi value. */
    double pi_max;         /**< Maximum observed Pi value. */
    double kmax;           /**< Maximum wavenumber represented in the analysis. */
} FluxMarks;

/**
 * @brief Persistent flux-lens band state and analysis buffers for a simulation context.
 *
 * The state owns band spectra, physical reconstructions, and bucket arrays after
 * flux_lens_ensure_capacity(); release them with flux_lens_release().
 */
typedef struct FluxLensState {
    bool enabled;              /**< True when flux-lens analysis is active. */
    bool locked;               /**< True when the current band should not chase new marks. */
    bool force_update;         /**< True when the next update should ignore period throttling. */
    bool band_ready;           /**< True when band_lo/band_hi describe an initialized band. */
    size_t field_index;        /**< Context field index analyzed by the lens. */
    double smoothing;          /**< Relaxation factor for target band updates. */
    double min_bandwidth;      /**< Minimum allowed spectral band width. */
    size_t update_period;      /**< Number of steps between automatic mark refreshes. */
    size_t last_update_step;   /**< Step index of the most recent mark refresh. */
    double band_lo;            /**< Active lower band edge. */
    double band_hi;            /**< Active upper band edge. */
    double target_band_lo;     /**< Desired lower band edge before relaxation. */
    double target_band_hi;     /**< Desired upper band edge before relaxation. */
    double band_max_component; /**< Maximum component magnitude inside reconstructed band. */
    double band_max_magnitude; /**< Maximum complex magnitude inside reconstructed band. */
    double max_k;              /**< Maximum represented wavenumber. */
    double complex *band_spec; /**< Owned spectral band buffer. */
    double complex *band_phys; /**< Owned physical reconstruction buffer. */
    size_t band_capacity;      /**< Allocated element capacity for band buffers. */
    double *S;                 /**< Owned spectral work-density array. */
    double *Pi;                /**< Owned cumulative flux array. */
    double *bucket_k;          /**< Owned bucket-center wavenumber array. */
    double *bucket_pi;         /**< Owned bucket flux array. */
    double *bucket_S;          /**< Owned bucket work-density array. */
    double *bucket_absS;       /**< Owned absolute work-density bucket array. */
    size_t bucket_capacity;    /**< Allocated element capacity for bucket arrays. */
    size_t bucket_count;       /**< Number of populated bucket entries. */
    double pi_min;             /**< Minimum Pi value from the latest analysis. */
    double pi_max;             /**< Maximum Pi value from the latest analysis. */
    double absS_total;         /**< Total absolute work-density from latest analysis. */
    FluxMarks marks;           /**< Latest spectral landmarks. */
    size_t scratch_bytes;      /**< Context-accounted bytes owned by this state. */
} FluxLensState;

/**
 * @brief Scratch workspace used while computing and reconstructing flux-lens bands.
 *
 * The workspace owns temporary buffers and its FFT plan after capacity is
 * ensured; release them with flux_release_scratch() or flux_lens_release().
 */
typedef struct FluxLensWorkspace {
    double complex *tmp_spec1; /**< Owned temporary spectral buffer. */
    double complex *tmp_phys1; /**< Owned temporary physical buffer. */
    double *tmp_real1;         /**< Owned temporary real-valued buffer. */
    bool use_dealias;          /**< True when dealiasing should be applied. */
    FFTPlan plan;              /**< Owned reusable FFT plan. */
    size_t capacity;           /**< Allocated element capacity for temporary buffers. */
    bool plan_ready;           /**< True when @ref plan has been initialized. */
    size_t scratch_bytes;      /**< Context-accounted bytes owned by this workspace. */
} FluxLensWorkspace;

void flux_lens_init(FluxLensState *lens);
void flux_lens_workspace_init(FluxLensWorkspace *workspace);
void flux_lens_force_refresh(FluxLensState *lens);
void flux_lens_set_field_index(FluxLensState *lens, size_t field_index);
void flux_lens_scale_width(FluxLensState *lens, double scale);
void flux_lens_shift_center(FluxLensState *lens, double shift);
void flux_lens_set_band(FluxLensState *lens, double band_lo, double band_hi);
SimResult flux_lens_ensure_capacity(FluxLensState *lens, FluxLensWorkspace *workspace,
                                    struct SimContext *context, size_t length);
void flux_lens_release(FluxLensState *lens, FluxLensWorkspace *workspace,
                       struct SimContext *context);

bool flux_compute(const struct SimField *field, FluxLensState *lens, FluxLensWorkspace *workspace,
                  struct SimContext *context);

void flux_lens_after_flux(FluxLensState *lens);
void flux_lens_update_targets_from_marks(FluxLensState *lens);
void flux_lens_relax_band(FluxLensState *lens);
bool flux_lens_reconstruct(const struct SimField *field, FluxLensState *lens,
                           FluxLensWorkspace *workspace);

void flux_lens_update(struct SimContext *context);

double flux_total_work(const double *S, size_t length);
double flux_total_energy(const double complex *u_hat, size_t length);

void flux_release_scratch(FluxLensWorkspace *workspace, struct SimContext *context);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_FLUX_LENS_H */
