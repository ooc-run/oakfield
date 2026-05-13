/**
 * @file field.h
 * @brief Multidimensional contiguous field abstraction with configurable layout.
 * @ingroup oakfield_fields
 *
 * @details SimField owns or views contiguous scalar storage with explicit
 * shape, stride, scalar-domain, and representation metadata. Constructors that
 * allocate storage transfer ownership to the field and require
 * sim_field_destroy(); wrapping constructors keep caller-owned data alive only
 * for the lifetime supplied by the caller.
 */
#ifndef OAKFIELD_FIELD_H
#define OAKFIELD_FIELD_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OAKFIELD_SIM_RESULT_DEFINED
#define OAKFIELD_SIM_RESULT_DEFINED
/**
 * @brief Return codes shared by libsimcore modules.
 */
typedef enum SimResult {
    SIM_RESULT_OK = 0,           /**< Operation succeeded. */
    SIM_RESULT_INVALID_ARGUMENT, /**< One or more arguments were invalid. */
    SIM_RESULT_INVALID_STATE,    /**< Operation not allowed in current state. */
    SIM_RESULT_OUT_OF_MEMORY,    /**< Allocation failed. */
    SIM_RESULT_NOT_FOUND,        /**< Requested element was not found. */
    SIM_RESULT_NOT_SUPPORTED,    /**< Requested capability is not supported. */
    SIM_RESULT_DEPENDENCY_ERROR, /**< Graph dependency resolution failed. */
    SIM_RESULT_TYPE_MISMATCH     /**< Type mismatch encountered. */
} SimResult;
#endif /* OAKFIELD_SIM_RESULT_DEFINED */

/**
 * @brief Supported field storage ordering schemes.
 */
typedef enum SimFieldStorage {
    SIM_FIELD_STORAGE_ROW_MAJOR = 0 /**< Row-major contiguous layout (C-style). */
} SimFieldStorage;

/**
 * @brief Domain for field representation (physical vs spectral).
 */
typedef enum SimFieldDomain {
    SIM_FIELD_DOMAIN_UNKNOWN = 0, /**< Domain has not been specified. */
    SIM_FIELD_DOMAIN_PHYSICAL,    /**< Samples live in physical/sample space. */
    SIM_FIELD_DOMAIN_SPECTRAL,    /**< Samples live in Fourier/spectral space. */
    SIM_FIELD_DOMAIN_HYBRID       /**< Representation mixes physical and spectral semantics. */
} SimFieldDomain;

/**
 * @brief Value kind for field scalars.
 */
typedef enum SimFieldValueKind {
    SIM_FIELD_VALUE_REAL_SCALAR = 0,         /**< Real-valued scalar storage. */
    SIM_FIELD_VALUE_COMPLEX_SCALAR,          /**< Complex-valued scalar storage. */
    SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT, /**< Complex storage with Hermitian symmetry / real
                                                signal. */
    SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT, /**< Complex storage with structurally zero
                                                     imaginary lane. */
    SIM_FIELD_VALUE_UNKNOWN                       /**< Value kind has not been specified. */
} SimFieldValueKind;

/**
 * @brief Authoritative representation description for a field.
 */
typedef struct SimFieldRepresentation {
    SimFieldDomain domain;        /**< Physical/spectral representation domain. */
    SimFieldValueKind value_kind; /**< Scalar value representation and constraints. */
} SimFieldRepresentation;

/**
 * @brief Scalar algebra domain used by IR typing and legality rules.
 *
 * This descriptor intentionally excludes representation constraints
 * (e.g., Hermitian symmetry), which remain in @ref SimFieldRepresentation.
 */
typedef enum SimScalarDomainKind {
    SIM_SCALAR_DOMAIN_UNKNOWN = 0, /**< Scalar domain is unspecified. */
    SIM_SCALAR_DOMAIN_REAL,        /**< Real floating-point algebra. */
    SIM_SCALAR_DOMAIN_COMPLEX,     /**< Complex floating-point algebra. */
    SIM_SCALAR_DOMAIN_INTEGER,     /**< Exact integer algebra. */
    SIM_SCALAR_DOMAIN_MODULAR      /**< Integer algebra modulo a fixed modulus. */
} SimScalarDomainKind;

/**
 * @brief Bitmask describing operations supported by a scalar domain.
 */
typedef enum SimScalarCapability {
    SIM_SCALAR_CAP_NONE = 0u,                           /**< No scalar operations are guaranteed. */
    SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC = 1u << 0,       /**< Addition/subtraction supported. */
    SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC = 1u << 1, /**< Multiplication supported. */
    SIM_SCALAR_CAP_DIVISION = 1u << 2,                  /**< Division supported. */
    SIM_SCALAR_CAP_ORDERING = 1u << 3,                  /**< Ordering comparisons supported. */
    SIM_SCALAR_CAP_FLOOR = 1u << 4,                     /**< Floor operation supported. */
    SIM_SCALAR_CAP_MODULO = 1u << 5,           /**< Modulo/remainder operation supported. */
    SIM_SCALAR_CAP_ANALYTIC_CALL = 1u << 6,    /**< Analytic function calls supported. */
    SIM_SCALAR_CAP_CONJUGATION = 1u << 7,      /**< Complex conjugation supported. */
    SIM_SCALAR_CAP_COMPLEX_ROTATION = 1u << 8, /**< Complex phase rotation supported. */
    SIM_SCALAR_CAP_POWER = 1u << 9             /**< Power operation supported. */
} SimScalarCapability;

/**
 * @brief Unified scalar-domain descriptor.
 *
 * `kind` encodes scalar algebra class. `bit_width`, `is_signed`, and `modulus`
 * provide parameters for integer/modular families.
 */
typedef struct SimScalarDomain {
    SimScalarDomainKind kind; /**< Scalar algebra family. */
    uint16_t bit_width;       /**< Width of underlying scalar lane (0 when unknown). */
    bool is_signed;           /**< Sign flag for integer/modular kinds. */
    uint64_t modulus;         /**< Modulus for modular arithmetic (0 when not modular). */
} SimScalarDomain;

struct SimField;

/**
 * @brief Allocation callbacks used by fields.
 */
typedef void *(*SimFieldAllocFn)(void *userdata, size_t size);

/**
 * @brief Deallocation callback used by fields.
 */
typedef void (*SimFieldFreeFn)(void *userdata, void *ptr);

/**
 * @brief Custom allocator description for field memory.
 */
typedef struct SimFieldAllocator {
    SimFieldAllocFn allocate; /**< Allocation callback, must behave like malloc. */
    SimFieldFreeFn release;   /**< Deallocation callback, must behave like free. */
    void *userdata;           /**< User-defined pointer passed to callbacks. */
} SimFieldAllocator;

/**
 * @brief Layout description for a field.
 */
typedef struct SimFieldLayout {
    size_t rank;     /**< Number of logical dimensions. */
    size_t *shape;   /**< Element counts per dimension (axis 0 slowest, axis rank-1 fastest). */
    size_t *strides; /**< Strides (in elements) per dimension for row-major layout. */
    bool contiguous; /**< True if the field is stored contiguously. */
} SimFieldLayout;

/**
 * @brief Owning multidimensional field.
 */
typedef struct SimField {
    SimFieldLayout layout;       /**< Layout metadata. */
    size_t element_size;         /**< Size of a single element in bytes. */
    void *data;                  /**< Contiguous data buffer. */
    SimFieldStorage storage;     /**< Storage ordering. */
    SimFieldAllocator allocator; /**< Allocator used for data buffer. */
    bool owns_data;              /**< Indicates whether the field owns @ref data. */
    bool complex_mode; /**< Deprecated legacy hint; use representation/scalar-domain APIs. */
    SimFieldRepresentation repr;   /**< Authoritative representation metadata. */
    SimScalarDomain scalar_domain; /**< Explicit scalar domain metadata for this field. */
    uint64_t magic;                /**< Internal guard to detect initialization. */
} SimField;

/**
 * @brief Explicit in-memory representation for a complex double scalar (re, im)
 *
 * Used instead of "double complex" in public headers to avoid C++/Objective-C++ parsing issues.
 */
typedef struct SimComplexDouble {
    double re; /**< Real component. */
    double im; /**< Imaginary component. */
} SimComplexDouble;

#ifndef SIM_HAVE_SIMCOMPLEXDOUBLE
#define SIM_HAVE_SIMCOMPLEXDOUBLE 1
#endif

/**
 * @brief Populate allocator with libc malloc/free.
 *
 * @param[out] allocator Target allocator record.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_field_allocator_default(SimFieldAllocator *allocator);

/**
 * @brief Initialize an owning field and allocate storage.
 *
 * @param[out] field Field instance to initialize.
 * @param rank Dimensionality of the field.
 * @param shape Array of @p rank entries describing dimension extents.
 * @param element_size Size of a single element in bytes.
 * @param storage Desired storage ordering.
 * @param allocator Optional allocator (pass NULL for default).
 * @return #SIM_RESULT_OK on success, error code otherwise.
 */
SimResult sim_field_init(SimField *field, size_t rank, const size_t *shape, size_t element_size,
                         SimFieldStorage storage, const SimFieldAllocator *allocator);

/**
 * @brief Initialize an owning field with an explicit scalar domain.
 *
 * This constructor avoids ambiguous size-based inference for integer fields.
 *
 * @param[out] field Field instance to initialize.
 * @param rank Dimensionality of the field.
 * @param shape Array of @p rank entries describing dimension extents.
 * @param scalar_domain Exact scalar domain to use for storage and semantics.
 * @param storage Desired storage ordering.
 * @param allocator Optional allocator (pass NULL for default).
 * @return #SIM_RESULT_OK on success, error code otherwise.
 */
SimResult sim_field_init_typed(SimField *field, size_t rank, const size_t *shape,
                               SimScalarDomain scalar_domain, SimFieldStorage storage,
                               const SimFieldAllocator *allocator);

/**
 * @brief Initialize a non-owning view over existing data.
 *
 * @param[out] field Field instance to initialize.
 * @param layout Layout metadata copied into the field.
 * @param element_size Size of a single element in bytes.
 * @param storage Storage ordering of @p data.
 * @param data Pointer to externally managed buffer.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_wrap(SimField *field, const SimFieldLayout *layout, size_t element_size,
                         SimFieldStorage storage, void *data);

/**
 * @brief Initialize a non-owning view with an explicit scalar domain.
 *
 * This constructor avoids ambiguous size-based inference for integer fields.
 *
 * @param[out] field Field instance to initialize.
 * @param layout Layout metadata copied into the field.
 * @param scalar_domain Exact scalar domain to use for storage and semantics.
 * @param storage Storage ordering of @p data.
 * @param data Pointer to externally managed buffer.
 * @return #SIM_RESULT_OK on success or an error code otherwise.
 */
SimResult sim_field_wrap_typed(SimField *field, const SimFieldLayout *layout,
                               SimScalarDomain scalar_domain, SimFieldStorage storage, void *data);

/**
 * @brief Release all resources held by a field.
 *
 * @param field Field to destroy; may be NULL.
 */
void sim_field_destroy(SimField *field);

/**
 * @brief Compute the linear offset for provided indices.
 *
 * @param field Field instance.
 * @param indices Array of length @ref SimFieldLayout::rank.
 * @param[out] out_offset Receives the byte offset within @ref SimField::data.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_index_offset(const SimField *field, const size_t *indices, size_t *out_offset);

/**
 * @brief Compute the linear element index for the provided indices.
 *
 * @param field Field instance.
 * @param indices Array of indices.
 * @param rank Number of indices supplied (must match field rank).
 * @param[out] out_index Receives the element index (not bytes).
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_element_index(const SimField *field, const size_t *indices, size_t rank,
                                  size_t *out_index);

/**
 * @brief Obtain the raw data pointer.
 *
 * @param field Field instance.
 * @return Writable data pointer, or NULL if uninitialized.
 */
void *sim_field_data(SimField *field);

/**
 * @brief Obtain the raw data pointer (const-qualified).
 *
 * @param field Field instance.
 * @return Read-only data pointer, or NULL if uninitialized.
 */
const void *sim_field_data_const(const SimField *field);

/**
 * @brief Returns the total number of scalar elements in the field layout.
 */
size_t sim_field_element_count(const SimFieldLayout *layout);

/**
 * @brief Compute the total number of bytes occupied by the field.
 *
 * @param field Field instance.
 * @return Total byte size, or 0 for an invalid field.
 */
size_t sim_field_bytes(const SimField *field);

/**
 * @brief Access the field rank.
 *
 * @param field Field instance.
 * @return Rank value; zero if @p field is NULL.
 */
size_t sim_field_rank(const SimField *field);

/**
 * @brief Access the shape array.
 *
 * @param field Field instance.
 * @return Pointer to the shape array, or NULL.
 */
const size_t *sim_field_shape(const SimField *field);

/**
 * @brief Access the stride array.
 *
 * @param field Field instance.
 * @return Pointer to the stride array, or NULL.
 */
const size_t *sim_field_strides(const SimField *field);

/**
 * @brief Convenience width accessor for the fastest-varying axis.
 *
 * For rank >= 1, returns shape[rank-1]. Returns 0 for invalid fields.
 */
size_t sim_field_width(const SimField *field);

/**
 * @brief Convenience height accessor for the next-to-fastest axis.
 *
 * For rank >= 2, returns shape[rank-2]. For rank == 1, returns 1.
 */
size_t sim_field_height(const SimField *field);

/**
 * @brief Convert a linear element index into 2D coordinates.
 *
 * Uses the canonical convention: x=axis rank-1 (fast), y=axis rank-2 (slow).
 * For rank == 1, y is set to 0. For rank > 2, other axes are assumed 0.
 */
SimResult sim_field_index_to_xy(const SimField *field, size_t index, size_t *out_x, size_t *out_y);

/**
 * @brief Convert 2D coordinates into a linear element index.
 *
 * Uses the canonical convention: x=axis rank-1 (fast), y=axis rank-2 (slow).
 * For rank == 1, y must be 0. For rank > 2, other axes are assumed 0.
 */
SimResult sim_field_xy_to_index(const SimField *field, size_t x, size_t y, size_t *out_index);

/**
 * @brief Returns true if the field must behave as complex.
 *
 * True when the field is already stored as complex or has been marked to
 * require complex storage (the runtime will promote if needed).
 */
bool sim_field_complex_mode(const SimField *field);

/**
 * @brief Returns whether the field storage is complex (`SimComplexDouble`).
 *
 * @param field Field instance.
 * @return true when element size equals 2*sizeof(double), false otherwise.
 */
bool sim_field_storage_is_complex(const SimField *field);

/**
 * @brief Returns whether the field scalar-domain semantics are complex.
 *
 * This is representation/scalar-domain aware and may differ from raw storage during
 * transitional states. Prefer this for semantic dispatch.
 */
bool sim_field_domain_is_complex(const SimField *field);

/**
 * @brief Legacy compatibility alias for complex-storage checks.
 *
 * Deprecated for semantic decisions; use `sim_field_domain_is_complex(...)` for algebraic
 * behavior, or `sim_field_storage_is_complex(...)` for memory-layout checks.
 */
bool sim_field_is_complex(const SimField *field);

/**
 * @brief Returns the number of scalar components per logical element.
 *
 * For standard double-precision scalars returns 1, for complex double returns 2.
 */
size_t sim_field_components(const SimField *field);

/**
 * @brief Convenience accessor for fields with scalar double elements.
 *
 * @param field Field instance.
 * @return Pointer to the data as a double array, or NULL if field is invalid.
 */
double *sim_field_real_data(SimField *field);
const double *sim_field_real_data_const(const SimField *field);
int8_t *sim_field_i8_data(SimField *field);
const int8_t *sim_field_i8_data_const(const SimField *field);
int32_t *sim_field_i32_data(SimField *field);
const int32_t *sim_field_i32_data_const(const SimField *field);
int64_t *sim_field_i64_data(SimField *field);
const int64_t *sim_field_i64_data_const(const SimField *field);
uint8_t *sim_field_u8_data(SimField *field);
const uint8_t *sim_field_u8_data_const(const SimField *field);
uint32_t *sim_field_u32_data(SimField *field);
const uint32_t *sim_field_u32_data_const(const SimField *field);
uint64_t *sim_field_u64_data(SimField *field);
const uint64_t *sim_field_u64_data_const(const SimField *field);

/**
 * @brief Convenience accessor for fields with complex double elements.
 *
 * Interpret backing memory as an array of double complex (re,im); returns NULL when
 * the field doesn't correspond to complex double storage.
 */
SimComplexDouble *sim_field_complex_data(SimField *field);
const SimComplexDouble *sim_field_complex_data_const(const SimField *field);

SimResult sim_field_promote_inplace_to_complex(SimField *field);

/**
 * @brief Force a field into complex storage if it is not already using it.
 *
 * Safe to call repeatedly; returns an error if promotion is not possible.
 */
SimResult sim_field_require_complex(SimField *field);
SimFieldRepresentation sim_field_representation(const SimField *field);
SimScalarDomain sim_field_scalar_domain(const SimField *field);
SimResult sim_field_set_scalar_domain(SimField *field, SimScalarDomain domain);
bool sim_field_storage_matches_scalar_domain(size_t element_size, SimScalarDomain domain);
SimScalarDomain sim_scalar_domain_unknown(void);
SimScalarDomain sim_scalar_domain_f64(void);
SimScalarDomain sim_scalar_domain_c64(void);
SimScalarDomain sim_scalar_domain_i8(void);
SimScalarDomain sim_scalar_domain_i32(void);
SimScalarDomain sim_scalar_domain_i64(void);
SimScalarDomain sim_scalar_domain_u8(void);
SimScalarDomain sim_scalar_domain_u32(void);
SimScalarDomain sim_scalar_domain_u64(void);
SimScalarDomain sim_scalar_domain_from_legacy_complex_flag(bool is_complex);
SimScalarDomain sim_scalar_domain_from_field_representation(SimFieldRepresentation repr);
SimScalarDomain sim_scalar_domain_from_field(const SimField *field);
const char *sim_scalar_domain_kind_name(SimScalarDomainKind kind);
bool sim_scalar_domain_kind_from_name(const char *text, SimScalarDomainKind *out_kind);
const char *sim_scalar_domain_name(SimScalarDomain domain);
bool sim_scalar_domain_from_name(const char *text, SimScalarDomain *out_domain);
bool sim_scalar_domain_equal(SimScalarDomain lhs, SimScalarDomain rhs);
bool sim_scalar_domain_validate(SimScalarDomain domain);
bool sim_scalar_domain_is_complex(SimScalarDomain domain);
bool sim_scalar_domain_is_integer(SimScalarDomain domain);
uint32_t sim_scalar_domain_capabilities(SimScalarDomain domain);
bool sim_scalar_domain_supports(SimScalarDomain domain, uint32_t capability_mask);
const char *sim_field_domain_name(SimFieldDomain domain);
const char *sim_field_value_kind_name(SimFieldValueKind kind);
bool sim_field_domain_from_name(const char *text, SimFieldDomain *out_domain);
bool sim_field_value_kind_from_name(const char *text, SimFieldValueKind *out_kind);
bool sim_field_value_kind_is_complex_valued(SimFieldValueKind kind);
bool sim_field_value_kind_has_imag_zero_constraint(SimFieldValueKind kind);
bool sim_field_value_kind_has_spectral_real_constraint(SimFieldValueKind kind);
bool sim_field_representation_requires_complex_storage(SimFieldRepresentation repr);
bool sim_field_representation_has_imag_zero_constraint(SimFieldRepresentation repr);
bool sim_field_representation_has_spectral_real_constraint(SimFieldRepresentation repr);
SimResult sim_field_validate_representation(const SimField *field, SimFieldRepresentation repr);
SimResult sim_field_set_representation(SimField *field, SimFieldRepresentation repr);
#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_FIELD_H */
