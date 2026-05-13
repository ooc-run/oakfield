/**
 * @file operator_utils.h
 * @brief Utilities shared by operator implementations (naming, clamping, math helpers).
 */
#ifndef OAKFIELD_OPERATOR_UTILS_H
#define OAKFIELD_OPERATOR_UTILS_H

#include <stddef.h>
#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Produce a unique operator name by appending an integer suffix.
 *
 * The buffer is guaranteed to remain null-terminated when capacity is nonzero.
 * Truncated strings may occur for small buffers.
 *
 * @param buffer Destination character buffer.
 * @param capacity Size of @p buffer in bytes including the null terminator slot.
 * @param prefix Base prefix to use when generating the label.
 */
void sim_operator_make_unique_name(char* buffer, size_t capacity, const char* prefix);

/**
 * @brief Resolve scale_by_dt with defaults and warnings.
 *
 * @param context Simulation context used for time-model checks.
 * @param op_name Human-readable operator label for warnings.
 * @param has_user_value True if the caller supplied a value; false to use default.
 * @param requested Value requested by the caller; ignored when has_user_value is false.
 * @return Resolved scale_by_dt value; defaults to true when no user value is supplied.
 */
bool sim_operator_resolve_scale_by_dt(const struct SimContext* context,
                                      const char*              op_name,
                                      bool                     has_user_value,
                                      bool                     requested);

/**
 * @brief Resolve the requested time mode under the active representation policy.
 *
 * @param context Simulation context.
 * @param op_config Optional operator config for representation override.
 * @param requested Requested clock mode.
 * @param nominal_dt Nominal dt used for step-based pure time.
 * @param epsilon Threshold below which nominal_dt is treated as unset.
 * @param[out] out_forced_pure Optional flag set true when a pure clock is forced.
 * @return Resolved clock mode; strict policies may coerce accumulated clocks.
 */
SimClockMode sim_operator_choose_time_mode(const struct SimContext* context,
                                           const SimOperatorConfig* op_config,
                                           SimClockMode             requested,
                                           double                   nominal_dt,
                                           double                   epsilon,
                                           bool*                    out_forced_pure);

/**
 * @brief Return true when kernel-backed execution is permitted by policy.
 *
 * @param context Simulation context supplying backend and determinism policy.
 * @param op_config Optional operator-level backend override.
 * @param required_features Backend feature mask required by the kernel.
 * @param determinism_flags Determinism properties required by the kernel.
 * @return true when the kernel path may be registered; false for split fallback.
 */
bool sim_operator_should_register_kernel(const struct SimContext* context,
                                         const SimOperatorConfig* op_config,
                                         uint64_t                 required_features,
                                         SimDeterminismFlags      determinism_flags);

/**
 * @brief Kernel registration policy with schema-level stabilization gating.
 *
 * Certain legacy operator schemas remain split-only by default while parity work
 * is in progress. Set OAKFIELD_ENABLE_EXPERIMENTAL_LEGACY_KERNELS=1 to bypass
 * this guard for local experiments.
 *
 * @param context Simulation context supplying backend and determinism policy.
 * @param op_config Optional operator-level backend override.
 * @param required_features Backend feature mask required by the kernel.
 * @param determinism_flags Determinism properties required by the kernel.
 * @param schema_key Stable schema key used by the stabilization allow/deny list.
 * @return true when the schema may register a kernel; false for split fallback.
 */
bool sim_operator_should_register_kernel_for_schema(const struct SimContext* context,
                                                    const SimOperatorConfig* op_config,
                                                    uint64_t                 required_features,
                                                    SimDeterminismFlags      determinism_flags,
                                                    const char*              schema_key);

/**
 * @brief Return true when @p field carries exact f64 scalar-domain metadata.
 *
 * @param field Field to inspect.
 * @return true for f64 scalar-domain fields; false for NULL or other domains.
 */
bool sim_operator_field_domain_is_f64(const SimField* field);

/**
 * @brief Return true when @p field carries f64 or c64 scalar-domain metadata.
 *
 * @param field Field to inspect.
 * @return true for f64/c64 scalar-domain fields; false for NULL or other domains.
 */
bool sim_operator_field_domain_is_f64_or_c64(const SimField* field);

/**
 * @brief Return true when @p domain is a supported exact integer field domain.
 *
 * @param domain Scalar domain to inspect.
 * @return true for supported signed or unsigned integer domains.
 */
bool sim_operator_domain_is_exact_integer(SimScalarDomain domain);

/**
 * @brief Truncate a raw integer value to the active width of @p domain.
 *
 * @param raw Raw integer bits.
 * @param domain Target integer scalar domain.
 * @return Raw value masked to the storage width of @p domain.
 */
uint64_t sim_operator_integer_truncate(uint64_t raw, SimScalarDomain domain);

/**
 * @brief Interpret a truncated raw integer in @p domain as a signed 64-bit value.
 *
 * @param raw Raw integer bits in the field domain.
 * @param domain Integer scalar domain.
 * @return Sign-extended signed interpretation for signed domains, or cast value otherwise.
 */
int64_t sim_operator_integer_as_i64(uint64_t raw, SimScalarDomain domain);

/**
 * @brief Convert a signed literal into raw storage for @p domain.
 *
 * @param value Signed integer literal.
 * @param domain Target integer scalar domain.
 * @param[out] out_raw Receives truncated raw storage bits.
 * @return true when @p domain is supported and @p out_raw is non-NULL.
 */
bool sim_operator_integer_raw_from_signed(int64_t value, SimScalarDomain domain, uint64_t* out_raw);

/**
 * @brief Convert an unsigned literal into raw storage for @p domain.
 *
 * @param value Unsigned integer literal.
 * @param domain Target integer scalar domain.
 * @param[out] out_raw Receives truncated raw storage bits.
 * @return true when @p domain is supported and @p out_raw is non-NULL.
 */
bool sim_operator_integer_raw_from_unsigned(uint64_t value,
                                            SimScalarDomain domain,
                                            uint64_t*       out_raw);

/**
 * @brief Convert an exactly representable integer-valued double into raw storage for @p domain.
 *
 * This intentionally rejects non-integral values and large integers that cannot
 * be represented exactly by IEEE-754 doubles. Use raw integer overrides for
 * larger literals.
 *
 * @param value Integer-valued double literal.
 * @param domain Target integer scalar domain.
 * @param[out] out_raw Receives truncated raw storage bits.
 * @return true when conversion is exact and @p domain is supported.
 */
bool sim_operator_integer_raw_from_double(double value, SimScalarDomain domain, uint64_t* out_raw);

/**
 * @brief Compare two raw integer values under @p domain ordering semantics.
 *
 * @param lhs_raw Left raw integer value.
 * @param rhs_raw Right raw integer value.
 * @param domain Integer scalar domain controlling signedness.
 * @return -1 when lhs < rhs, 0 when lhs == rhs, 1 when lhs > rhs.
 */
int sim_operator_integer_compare(uint64_t lhs_raw, uint64_t rhs_raw, SimScalarDomain domain);

/**
 * @brief Read an integer element from a field-aligned buffer as raw storage.
 *
 * @param data Pointer to the field data buffer.
 * @param domain Integer scalar domain describing element width and signedness.
 * @param index Element index to read.
 * @param[out] out_raw Receives raw storage bits.
 * @return true on success; false for NULL pointers or unsupported domains.
 */
bool sim_operator_integer_read(const void* data,
                               SimScalarDomain domain,
                               size_t          index,
                               uint64_t*       out_raw);

/**
 * @brief Write an integer raw value into a field-aligned buffer.
 *
 * @param data Pointer to the field data buffer.
 * @param domain Integer scalar domain describing element width and signedness.
 * @param index Element index to write.
 * @param raw Raw value; truncated to @p domain before storage.
 * @return true on success; false for NULL data or unsupported domains.
 */
bool sim_operator_integer_write(void* data, SimScalarDomain domain, size_t index, uint64_t raw);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_OPERATOR_UTILS_H */
