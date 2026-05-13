/**
 * @file field.c
 * @brief Owning and non-owning multidimensional field storage implementation.
 *
 * This module manages SimField allocation, wrapping, scalar-domain metadata,
 * representation validation, and index/shape conversion. Storage is contiguous
 * row-major memory for owning fields, with explicit scalar domains used to keep
 * real, complex, and integer-valued fields aligned with KernelIR legality rules.
 */
#include "oakfield/field.h"
#include "oakfield/sim_buffer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <complex.h>
#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif
#if defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#define SIM_FIELD_MAGIC UINT64_C(0x53494D4649454C44) /* "SIMFIELD" */
#define SIM_FIELD_DEFAULT_ALIGNMENT 64U

static bool sim_field_is_initialized(const SimField* field) {
    return field != NULL && field->magic == SIM_FIELD_MAGIC;
}

static bool sim_field_text_iequals(const char* a, const char* b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return (*a == '\0') && (*b == '\0');
}

static SimScalarDomain sim_scalar_domain_make(SimScalarDomainKind kind,
                                              uint16_t            bit_width,
                                              bool                is_signed,
                                              uint64_t            modulus) {
    SimScalarDomain domain;
    domain.kind      = kind;
    domain.bit_width = bit_width;
    domain.is_signed = is_signed;
    domain.modulus   = modulus;
    return domain;
}

static SimScalarDomain sim_field_default_scalar_domain(size_t element_size) {
    return sim_buffer_data_type_to_scalar_domain(
        sim_buffer_data_type_from_element_size(element_size));
}

static size_t sim_field_element_size_for_scalar_domain(SimScalarDomain domain) {
    return sim_buffer_element_size(sim_buffer_data_type_from_scalar_domain(domain));
}

bool sim_field_storage_matches_scalar_domain(size_t element_size, SimScalarDomain domain) {
    if (element_size == 0U || !sim_scalar_domain_validate(domain)) {
        return false;
    }

    switch (domain.kind) {
        case SIM_SCALAR_DOMAIN_UNKNOWN:
            return true;
        case SIM_SCALAR_DOMAIN_REAL:
            return element_size == sizeof(double);
        case SIM_SCALAR_DOMAIN_COMPLEX:
            return element_size == sizeof(SimComplexDouble);
        case SIM_SCALAR_DOMAIN_INTEGER:
        case SIM_SCALAR_DOMAIN_MODULAR:
            return domain.bit_width == element_size * 8U;
        default:
            return false;
    }
}

static SimScalarDomain sim_field_storage_scalar_domain(const SimField* field) {
    if (field == NULL) {
        return sim_scalar_domain_unknown();
    }
    return sim_field_default_scalar_domain(field->element_size);
}

static SimScalarDomain sim_field_promotion_source_domain(const SimField* field) {
    SimFieldRepresentation repr;
    SimScalarDomain       domain;

    if (!sim_field_is_initialized(field)) {
        return sim_scalar_domain_unknown();
    }

    repr   = sim_field_representation(field);
    domain = sim_scalar_domain_from_field_representation(repr);
    if (domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN) {
        return domain;
    }
    return sim_field_storage_scalar_domain(field);
}

static void      sim_field_reset(SimField* field);
static SimResult sim_field_layout_init(SimFieldLayout* layout, size_t rank, const size_t* shape);

static void* sim_default_malloc(void* userdata, size_t size) {
    (void) userdata;
#if defined(__APPLE__) || defined(_POSIX_VERSION)
    void* ptr = NULL;
    if (size == 0U) {
        size = 1U;
    }
    if (posix_memalign(&ptr, SIM_FIELD_DEFAULT_ALIGNMENT, size) == 0) {
        return ptr;
    }
    return NULL;
#else
    return malloc(size);
#endif
}

static void sim_default_free(void* userdata, void* ptr) {
    (void) userdata;
    free(ptr);
}

static void sim_field_layout_reset(SimFieldLayout* layout) {
    if (layout == NULL) {
        return;
    }
#if defined(__APPLE__)
    if (layout->shape && malloc_size(layout->shape) == 0U) {
        layout->shape = NULL;
    }
    if (layout->strides && malloc_size(layout->strides) == 0U) {
        layout->strides = NULL;
    }
#endif
    free(layout->shape);
    free(layout->strides);
    layout->shape      = NULL;
    layout->strides    = NULL;
    layout->rank       = 0U;
    layout->contiguous = false;
}

SimResult sim_field_allocator_default(SimFieldAllocator* allocator) {
    if (allocator == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    allocator->allocate = sim_default_malloc;
    allocator->release  = sim_default_free;
    allocator->userdata = NULL;
    return SIM_RESULT_OK;
}

static SimResult sim_field_layout_init(SimFieldLayout* layout, size_t rank, const size_t* shape) {
    size_t i;

    if (layout == NULL || shape == NULL || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    layout->rank       = rank;
    layout->shape      = (size_t*) calloc(rank, sizeof(size_t));
    layout->strides    = (size_t*) calloc(rank, sizeof(size_t));
    layout->contiguous = false;
    if (layout->shape == NULL || layout->strides == NULL) {
        sim_field_layout_reset(layout);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (i = 0; i < rank; ++i) {
        layout->shape[i] = shape[i];
    }

    if (rank > 0U) {
        size_t stride = 1U;
        for (i = rank; i-- > 0U;) {
            layout->strides[i] = stride;
            stride *= (shape[i] > 0U) ? shape[i] : 1U;
        }
        layout->contiguous = true;
    }

    return SIM_RESULT_OK;
}

/* Count total elements from layout */
size_t sim_field_element_count(const SimFieldLayout* layout) {
    if (!layout || layout->rank == 0)
        return 0;
    size_t n = 1;
    for (size_t i = 0; i < layout->rank; ++i) {
        size_t d = layout->shape[i];
        if (d == 0)
            return 0;
        n *= d;
    }
    return n;
}

static void sim_field_reset(SimField* field) {
    if (field == NULL) {
        return;
    }

    const bool initialized = sim_field_is_initialized(field);

    if (initialized && field->owns_data && field->data != NULL &&
        field->allocator.release != NULL) {
#if defined(__APPLE__)
        if (field->allocator.release == sim_default_free && malloc_size(field->data) == 0U) {
            field->data = NULL;
        }
#endif
        if (field->data != NULL) {
            field->allocator.release(field->allocator.userdata, field->data);
        }
    }

    if (initialized) {
        sim_field_layout_reset(&field->layout);
    } else {
        field->layout.rank       = 0U;
        field->layout.shape      = NULL;
        field->layout.strides    = NULL;
        field->layout.contiguous = false;
    }

    field->data               = NULL;
    field->element_size       = 0U;
    field->storage            = SIM_FIELD_STORAGE_ROW_MAJOR;
    field->allocator.allocate = NULL;
    field->allocator.release  = NULL;
    field->allocator.userdata = NULL;
    field->owns_data          = false;
    field->complex_mode       = false;
    field->repr.domain        = SIM_FIELD_DOMAIN_UNKNOWN;
    field->repr.value_kind    = SIM_FIELD_VALUE_UNKNOWN;
    field->scalar_domain      = sim_scalar_domain_unknown();
    field->magic              = 0U;
}

void sim_field_destroy(SimField* field) {
    sim_field_reset(field);
}

/* Promote contiguous double-backed storage to complex<double> (re, im).
 * If it’s already complex, this is a no-op.
 * If it’s real (element_size == sizeof(double)), we allocate a new buffer,
 * copy real -> .re and set .im = 0, then free the old one if the field owns it.
 * Non-contiguous layouts and non-double element sizes are rejected.
 *
 * Post-condition:
 *   - field->element_size == sizeof(SimComplexDouble)
 *   - field->data points to complex buffer
 */
SimResult sim_field_promote_inplace_to_complex(SimField* field) {
    SimScalarDomain       source_domain;
    SimFieldRepresentation repr;

    if (!sim_field_is_initialized(field) || field->data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->element_size == sizeof(SimComplexDouble)) {
        /* Already complex */
        return SIM_RESULT_OK;
    }

    const size_t n = sim_field_element_count(&field->layout);
    if (n == 0)
        return SIM_RESULT_OK; /* nothing to do */

    if (!field->layout.contiguous) {
        return SIM_RESULT_INVALID_STATE; /* avoid interpreting non-contiguous layouts */
    }

    source_domain = sim_field_promotion_source_domain(field);
    if (!(source_domain.kind == SIM_SCALAR_DOMAIN_REAL ||
          source_domain.kind == SIM_SCALAR_DOMAIN_COMPLEX ||
          source_domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN)) {
        return SIM_RESULT_DEPENDENCY_ERROR; /* unsupported source algebra */
    }

    /* Conversion implementation is still double->c64 in v1. */
    if (field->element_size != sizeof(double)) {
        return SIM_RESULT_DEPENDENCY_ERROR; /* unsupported source storage */
    }

    /* Allocate complex buffer */
    size_t bytes    = n * sizeof(SimComplexDouble);
    void*  new_data = NULL;

    SimFieldAllocator al = field->allocator;
    if (!al.allocate) {
        al.allocate = sim_default_malloc;
        al.release  = sim_default_free;
    }
    new_data = al.allocate(al.userdata, bytes);
    if (!new_data)
        return SIM_RESULT_OUT_OF_MEMORY;

    /* Convert: real -> complex(re=real[i], im=0) */
    double* src = (double*) field->data;

#if defined(SIM_HAVE_VDSP)
    /* Interleaved copy + clear imag via vDSP when available. */
    double* dst_interleaved = (double*) new_data;
    double  one_scalar      = 1.0;
    vDSP_vsmulD(src, 1, &one_scalar, dst_interleaved, 2, n); /* copy real values into even slots */
    vDSP_vclrD(dst_interleaved + 1, 2, n);                   /* zero imaginary slots */
#else
    /* Scalar fallback */
    SimComplexDouble* dst = (SimComplexDouble*) new_data;
    for (size_t i = 0; i < n; ++i) {
        dst[i].re = src[i];
        dst[i].im = 0.0;
    }
#endif

    /* Replace buffer */
    if (field->owns_data && field->data) {
        if (!field->allocator.release)
            field->allocator.release = sim_default_free;
        field->allocator.release(field->allocator.userdata, field->data);
    }
    field->data            = new_data;
    field->element_size    = sizeof(SimComplexDouble);
    repr = sim_field_representation(field);
    if (repr.domain == SIM_FIELD_DOMAIN_UNKNOWN) {
        repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    }
    if (!sim_field_value_kind_is_complex_valued(repr.value_kind)) {
        repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    field->repr         = repr;
    field->complex_mode = sim_field_representation_requires_complex_storage(repr);
    field->scalar_domain = sim_scalar_domain_c64();
    field->owns_data = true; /* We own the new buffer */
    field->magic     = SIM_FIELD_MAGIC;

    return SIM_RESULT_OK;
}

SimResult sim_field_init(SimField*                field,
                         size_t                   rank,
                         const size_t*            shape,
                         size_t                   element_size,
                         SimFieldStorage          storage,
                         const SimFieldAllocator* allocator) {
    SimResult         result;
    size_t            total_elements;
    size_t            total_bytes;
    SimFieldAllocator local_allocator;

    if (field == NULL || shape == NULL || element_size == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    sim_field_reset(field);

    result = sim_field_layout_init(&field->layout, rank, shape);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (allocator == NULL) {
        result = sim_field_allocator_default(&local_allocator);
        if (result != SIM_RESULT_OK) {
            sim_field_destroy(field);
            return result;
        }
        field->allocator = local_allocator;
    } else {
        field->allocator = *allocator;
    }

    if (field->allocator.allocate == NULL || field->allocator.release == NULL) {
        sim_field_destroy(field);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    total_elements = sim_field_element_count(&field->layout);
    if (total_elements == 0U) {
        sim_field_destroy(field);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (element_size > 0U && total_elements > (SIZE_MAX / element_size)) {
        sim_field_destroy(field);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    total_bytes = total_elements * element_size;

    field->data = field->allocator.allocate(field->allocator.userdata, total_bytes);
    
    if (field->data == NULL) {
        sim_field_destroy(field);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    memset(field->data, 0, total_bytes);

    field->element_size = element_size;
    field->storage      = storage;
    field->complex_mode = false;
    field->repr.domain  = SIM_FIELD_DOMAIN_PHYSICAL;
    field->repr.value_kind =
        (element_size == sizeof(SimComplexDouble)) ? SIM_FIELD_VALUE_COMPLEX_SCALAR
                                                   : SIM_FIELD_VALUE_REAL_SCALAR;
    field->scalar_domain = sim_field_default_scalar_domain(element_size);
    field->owns_data = true;
    field->magic     = SIM_FIELD_MAGIC;

    return SIM_RESULT_OK;
}

SimResult sim_field_init_typed(SimField*                field,
                               size_t                   rank,
                               const size_t*            shape,
                               SimScalarDomain          scalar_domain,
                               SimFieldStorage          storage,
                               const SimFieldAllocator* allocator) {
    SimResult result;
    size_t    element_size = 0U;

    if (!sim_scalar_domain_validate(scalar_domain) ||
        scalar_domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    element_size = sim_field_element_size_for_scalar_domain(scalar_domain);
    if (element_size == 0U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    result = sim_field_init(field, rank, shape, element_size, storage, allocator);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_field_set_scalar_domain(field, scalar_domain);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(field);
        return result;
    }
    return SIM_RESULT_OK;
}

SimResult sim_field_wrap(SimField*             field,
                         const SimFieldLayout* layout,
                         size_t                element_size,
                         SimFieldStorage       storage,
                         void*                 data) {
    SimResult result;

    if (field == NULL || layout == NULL || layout->shape == NULL || layout->strides == NULL ||
        layout->rank == 0U || data == NULL || element_size == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    sim_field_reset(field);

    result = sim_field_layout_init(&field->layout, layout->rank, layout->shape);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (field->layout.strides != NULL) {
        memcpy(field->layout.strides, layout->strides, layout->rank * sizeof(size_t));
    }
    field->layout.contiguous = layout->contiguous;

    field->data         = data;
    field->element_size = element_size;
    field->storage      = storage;
    field->complex_mode = false;
    field->repr.domain  = SIM_FIELD_DOMAIN_PHYSICAL;
    field->repr.value_kind =
        (element_size == sizeof(SimComplexDouble)) ? SIM_FIELD_VALUE_COMPLEX_SCALAR
                                                   : SIM_FIELD_VALUE_REAL_SCALAR;
    field->scalar_domain = sim_field_default_scalar_domain(element_size);
    field->owns_data = false;
    field->magic     = SIM_FIELD_MAGIC;

    return SIM_RESULT_OK;
}

SimResult sim_field_wrap_typed(SimField*             field,
                               const SimFieldLayout* layout,
                               SimScalarDomain       scalar_domain,
                               SimFieldStorage       storage,
                               void*                 data) {
    SimResult result;
    size_t    element_size = 0U;

    if (!sim_scalar_domain_validate(scalar_domain) ||
        scalar_domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    element_size = sim_field_element_size_for_scalar_domain(scalar_domain);
    if (element_size == 0U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    result = sim_field_wrap(field, layout, element_size, storage, data);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_field_set_scalar_domain(field, scalar_domain);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(field);
        return result;
    }
    return SIM_RESULT_OK;
}

SimResult sim_field_index_offset(const SimField* field, const size_t* indices, size_t* out_offset) {
    size_t    elem_index = 0U;
    SimResult rc         = sim_field_element_index(
        field, indices, field != NULL ? field->layout.rank : 0U, &elem_index);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    *out_offset = elem_index * field->element_size;
    return SIM_RESULT_OK;
}

SimResult sim_field_element_index(const SimField* field,
                                  const size_t*   indices,
                                  size_t          rank,
                                  size_t*         out_index) {
    size_t offset = 0U;

    if (field == NULL || indices == NULL || out_index == NULL || field->layout.shape == NULL ||
        field->layout.strides == NULL || rank != field->layout.rank || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < rank; ++i) {
        if (indices[i] >= field->layout.shape[i]) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        offset += indices[i] * field->layout.strides[i];
    }

    *out_index = offset;
    return SIM_RESULT_OK;
}

void* sim_field_data(SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    return field->data;
}

const void* sim_field_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    return field->data;
}

size_t sim_field_bytes(const SimField* field) {
    size_t count;

    if (!sim_field_is_initialized(field)) {
        return 0U;
    }

    count = sim_field_element_count(&field->layout);
    return count * field->element_size;
}

size_t sim_field_rank(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return 0U;
    }
    return field->layout.rank;
}

const size_t* sim_field_shape(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    return field->layout.shape;
}

const size_t* sim_field_strides(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    return field->layout.strides;
}

size_t sim_field_width(const SimField* field) {
    if (!sim_field_is_initialized(field) || field->layout.shape == NULL ||
        field->layout.rank == 0U) {
        return 0U;
    }
    return field->layout.shape[field->layout.rank - 1U];
}

size_t sim_field_height(const SimField* field) {
    if (!sim_field_is_initialized(field) || field->layout.shape == NULL ||
        field->layout.rank == 0U) {
        return 0U;
    }
    if (field->layout.rank == 1U) {
        return 1U;
    }
    return field->layout.shape[field->layout.rank - 2U];
}

SimResult sim_field_index_to_xy(const SimField* field, size_t index, size_t* out_x, size_t* out_y) {
    if (!sim_field_is_initialized(field) || out_x == NULL || out_y == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->layout.shape == NULL || field->layout.strides == NULL || field->layout.rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&field->layout);
    if (count == 0U || index >= count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t rank = field->layout.rank;
    if (rank == 1U) {
        *out_x = index;
        *out_y = 0U;
        return SIM_RESULT_OK;
    }

    size_t axis_x   = rank - 1U;
    size_t axis_y   = rank - 2U;
    size_t extent_x = field->layout.shape[axis_x];
    size_t extent_y = field->layout.shape[axis_y];
    size_t stride_x = field->layout.strides[axis_x];
    size_t stride_y = field->layout.strides[axis_y];
    if (extent_x == 0U || extent_y == 0U || stride_x == 0U || stride_y == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_x = (index / stride_x) % extent_x;
    *out_y = (index / stride_y) % extent_y;
    return SIM_RESULT_OK;
}

SimResult sim_field_xy_to_index(const SimField* field, size_t x, size_t y, size_t* out_index) {
    if (!sim_field_is_initialized(field) || out_index == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->layout.shape == NULL || field->layout.strides == NULL || field->layout.rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t rank     = field->layout.rank;
    size_t axis_x   = rank - 1U;
    size_t extent_x = field->layout.shape[axis_x];
    size_t stride_x = field->layout.strides[axis_x];
    if (extent_x == 0U || stride_x == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (rank == 1U) {
        if (y != 0U || x >= extent_x) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        *out_index = x * stride_x;
        return SIM_RESULT_OK;
    }

    size_t axis_y   = rank - 2U;
    size_t extent_y = field->layout.shape[axis_y];
    size_t stride_y = field->layout.strides[axis_y];
    if (extent_y == 0U || stride_y == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (x >= extent_x || y >= extent_y) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_index = x * stride_x + y * stride_y;
    return SIM_RESULT_OK;
}

bool sim_field_value_kind_is_complex_valued(SimFieldValueKind kind) {
    return kind == SIM_FIELD_VALUE_COMPLEX_SCALAR ||
           kind == SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT ||
           kind == SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;
}

bool sim_field_value_kind_has_imag_zero_constraint(SimFieldValueKind kind) {
    return kind == SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;
}

bool sim_field_value_kind_has_spectral_real_constraint(SimFieldValueKind kind) {
    return kind == SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;
}

bool sim_field_representation_requires_complex_storage(SimFieldRepresentation repr) {
    return sim_field_value_kind_is_complex_valued(repr.value_kind);
}

bool sim_field_representation_has_imag_zero_constraint(SimFieldRepresentation repr) {
    return sim_field_value_kind_has_imag_zero_constraint(repr.value_kind);
}

bool sim_field_representation_has_spectral_real_constraint(SimFieldRepresentation repr) {
    return repr.domain == SIM_FIELD_DOMAIN_SPECTRAL &&
           sim_field_value_kind_has_spectral_real_constraint(repr.value_kind);
}

bool sim_field_complex_mode(const SimField* field) {
    SimFieldRepresentation repr;
    if (!sim_field_is_initialized(field)) {
        return false;
    }

    repr = sim_field_representation(field);
    if (sim_field_representation_requires_complex_storage(repr)) {
        return true;
    }

    /* Legacy bridge for callers that still set this flag directly. */
    return field->complex_mode;
}

bool sim_field_storage_is_complex(const SimField* f) {
    return sim_field_is_initialized(f) && (f->element_size == sizeof(SimComplexDouble));
}

bool sim_field_domain_is_complex(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return false;
    }
    if (sim_field_complex_mode(field)) {
        return true;
    }
    return sim_scalar_domain_is_complex(sim_scalar_domain_from_field(field));
}

/* Legacy compatibility alias: historically storage-based complexness query. */
bool sim_field_is_complex(const SimField* f) {
    return sim_field_storage_is_complex(f);
}

size_t sim_field_components(const SimField* field) {
    SimScalarDomain domain;

    if (!sim_field_is_initialized(field) || field->element_size == 0U) {
        return 0U;
    }
    domain = sim_scalar_domain_from_field(field);
    if (sim_scalar_domain_is_complex(domain)) {
        return 2U;
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER || domain.kind == SIM_SCALAR_DOMAIN_MODULAR) {
        return 1U;
    }
    if ((field->element_size % sizeof(double)) == 0U) {
        return field->element_size / sizeof(double);
    }
    return 1U;
}

double* sim_field_real_data(SimField* field) {
    SimScalarDomain domain;

    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    domain = sim_scalar_domain_from_field(field);
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER || domain.kind == SIM_SCALAR_DOMAIN_MODULAR ||
        sim_scalar_domain_is_complex(domain) || field->element_size == 0U ||
        (field->element_size % sizeof(double)) != 0U) {
        return NULL;
    }
    return (double*) field->data;
}

const double* sim_field_real_data_const(const SimField* field) {
    SimScalarDomain domain;

    if (!sim_field_is_initialized(field)) {
        return NULL;
    }
    domain = sim_scalar_domain_from_field(field);
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER || domain.kind == SIM_SCALAR_DOMAIN_MODULAR ||
        sim_scalar_domain_is_complex(domain) || field->element_size == 0U ||
        (field->element_size % sizeof(double)) != 0U) {
        return NULL;
    }
    return (const double*) field->data;
}

int8_t* sim_field_i8_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i8())) {
        return NULL;
    }
    return (int8_t*) field->data;
}

const int8_t* sim_field_i8_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i8())) {
        return NULL;
    }
    return (const int8_t*) field->data;
}

int32_t* sim_field_i32_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i32())) {
        return NULL;
    }
    return (int32_t*) field->data;
}

const int32_t* sim_field_i32_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i32())) {
        return NULL;
    }
    return (const int32_t*) field->data;
}

int64_t* sim_field_i64_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i64())) {
        return NULL;
    }
    return (int64_t*) field->data;
}

const int64_t* sim_field_i64_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_i64())) {
        return NULL;
    }
    return (const int64_t*) field->data;
}

uint8_t* sim_field_u8_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u8())) {
        return NULL;
    }
    return (uint8_t*) field->data;
}

const uint8_t* sim_field_u8_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u8())) {
        return NULL;
    }
    return (const uint8_t*) field->data;
}

uint32_t* sim_field_u32_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u32())) {
        return NULL;
    }
    return (uint32_t*) field->data;
}

const uint32_t* sim_field_u32_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u32())) {
        return NULL;
    }
    return (const uint32_t*) field->data;
}

uint64_t* sim_field_u64_data(SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u64())) {
        return NULL;
    }
    return (uint64_t*) field->data;
}

const uint64_t* sim_field_u64_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), sim_scalar_domain_u64())) {
        return NULL;
    }
    return (const uint64_t*) field->data;
}

SimComplexDouble* sim_field_complex_data(SimField* field) {
    if (!sim_field_is_initialized(field) || !sim_field_storage_is_complex(field)) {
        return NULL;
    }
    return (SimComplexDouble*) field->data;
}

const SimComplexDouble* sim_field_complex_data_const(const SimField* field) {
    if (!sim_field_is_initialized(field) || !sim_field_storage_is_complex(field)) {
        return NULL;
    }
    return (const SimComplexDouble*) field->data;
}

SimResult sim_field_require_complex(SimField* field) {
    SimScalarDomain       source_domain;
    SimFieldRepresentation repr;

    /* Preconditions:
     *   - field initialized, contiguous layout (non-contiguous promotion is rejected)
     *   - real fields must be double; complex already OK
     * Ownership/side effects:
     *   - promotes in-place by allocating a new complex buffer and freeing the old one if owned
     * Callers: split port binding, sim_context_prepare_complex_fields, misc operators.
     */
    if (!sim_field_is_initialized(field) || field->data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    source_domain = sim_field_promotion_source_domain(field);
    if (!(source_domain.kind == SIM_SCALAR_DOMAIN_REAL ||
          source_domain.kind == SIM_SCALAR_DOMAIN_COMPLEX ||
          source_domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN)) {
        return SIM_RESULT_DEPENDENCY_ERROR;
    }

    if (!sim_field_storage_is_complex(field)) {
        SimResult promote = sim_field_promote_inplace_to_complex(field);
        if (promote != SIM_RESULT_OK) {
            return promote;
        }
    }

    repr = sim_field_representation(field);
    if (repr.domain == SIM_FIELD_DOMAIN_UNKNOWN) {
        repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    }
    if (!sim_field_value_kind_is_complex_valued(repr.value_kind)) {
        repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    field->repr         = repr;
    field->complex_mode = sim_field_representation_requires_complex_storage(repr);
    field->scalar_domain = sim_scalar_domain_c64();
    return SIM_RESULT_OK;
}

SimFieldRepresentation sim_field_representation(const SimField* field) {
    SimFieldRepresentation repr = { .domain     = SIM_FIELD_DOMAIN_UNKNOWN,
                                    .value_kind = SIM_FIELD_VALUE_UNKNOWN };
    if (!sim_field_is_initialized(field)) {
        return repr;
    }
    repr = field->repr;
    /* If legacy flags are set, normalize representation to match underlying storage. */
    if (repr.domain == SIM_FIELD_DOMAIN_UNKNOWN) {
        repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    }
    if (repr.value_kind == SIM_FIELD_VALUE_REAL_SCALAR && sim_field_storage_is_complex(field)) {
        repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    return repr;
}

SimScalarDomain sim_field_scalar_domain(const SimField* field) {
    if (!sim_field_is_initialized(field)) {
        return sim_scalar_domain_unknown();
    }
    if (sim_scalar_domain_validate(field->scalar_domain) &&
        field->scalar_domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN) {
        return field->scalar_domain;
    }
    return sim_scalar_domain_from_field(field);
}

SimScalarDomain sim_scalar_domain_unknown(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_UNKNOWN, 0U, false, 0U);
}

SimScalarDomain sim_scalar_domain_f64(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_REAL, 64U, true, 0U);
}

SimScalarDomain sim_scalar_domain_c64(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_COMPLEX, 64U, true, 0U);
}

SimScalarDomain sim_scalar_domain_i8(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 8U, true, 0U);
}

SimScalarDomain sim_scalar_domain_i32(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 32U, true, 0U);
}

SimScalarDomain sim_scalar_domain_i64(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 64U, true, 0U);
}

SimScalarDomain sim_scalar_domain_u8(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 8U, false, 0U);
}

SimScalarDomain sim_scalar_domain_u32(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 32U, false, 0U);
}

SimScalarDomain sim_scalar_domain_u64(void) {
    return sim_scalar_domain_make(SIM_SCALAR_DOMAIN_INTEGER, 64U, false, 0U);
}

SimScalarDomain sim_scalar_domain_from_legacy_complex_flag(bool is_complex) {
    return is_complex ? sim_scalar_domain_c64() : sim_scalar_domain_f64();
}

SimScalarDomain sim_scalar_domain_from_field_representation(SimFieldRepresentation repr) {
    if (repr.value_kind == SIM_FIELD_VALUE_REAL_SCALAR) {
        return sim_scalar_domain_f64();
    }
    if (sim_field_value_kind_is_complex_valued(repr.value_kind)) {
        return sim_scalar_domain_c64();
    }
    return sim_scalar_domain_unknown();
}

SimScalarDomain sim_scalar_domain_from_field(const SimField* field) {
    SimScalarDomain       domain;
    SimFieldRepresentation repr;

    if (!sim_field_is_initialized(field)) {
        return sim_scalar_domain_unknown();
    }

    if (sim_scalar_domain_validate(field->scalar_domain) &&
        field->scalar_domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN) {
        return field->scalar_domain;
    }

    repr   = sim_field_representation(field);
    domain = sim_scalar_domain_from_field_representation(repr);
    if (domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN) {
        return domain;
    }

    if (field->complex_mode) {
        return sim_scalar_domain_c64();
    }

    if (field->element_size == sizeof(double)) {
        return sim_scalar_domain_f64();
    }
    if (field->element_size == sizeof(SimComplexDouble)) {
        return sim_scalar_domain_c64();
    }
    return sim_scalar_domain_unknown();
}

const char* sim_scalar_domain_kind_name(SimScalarDomainKind kind) {
    switch (kind) {
        case SIM_SCALAR_DOMAIN_REAL:
            return "real";
        case SIM_SCALAR_DOMAIN_COMPLEX:
            return "complex";
        case SIM_SCALAR_DOMAIN_INTEGER:
            return "integer";
        case SIM_SCALAR_DOMAIN_MODULAR:
            return "modular";
        case SIM_SCALAR_DOMAIN_UNKNOWN:
        default:
            return "unknown";
    }
}

bool sim_scalar_domain_kind_from_name(const char* text, SimScalarDomainKind* out_kind) {
    if (text == NULL || out_kind == NULL) {
        return false;
    }

    if (sim_field_text_iequals(text, "unknown")) {
        *out_kind = SIM_SCALAR_DOMAIN_UNKNOWN;
        return true;
    }
    if (sim_field_text_iequals(text, "real")) {
        *out_kind = SIM_SCALAR_DOMAIN_REAL;
        return true;
    }
    if (sim_field_text_iequals(text, "complex")) {
        *out_kind = SIM_SCALAR_DOMAIN_COMPLEX;
        return true;
    }
    if (sim_field_text_iequals(text, "integer")) {
        *out_kind = SIM_SCALAR_DOMAIN_INTEGER;
        return true;
    }
    if (sim_field_text_iequals(text, "modular")) {
        *out_kind = SIM_SCALAR_DOMAIN_MODULAR;
        return true;
    }
    return false;
}

const char* sim_scalar_domain_name(SimScalarDomain domain) {
    if (domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        return "unknown";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_REAL && domain.bit_width == 64U) {
        return "f64";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_COMPLEX && domain.bit_width == 64U) {
        return "c64";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 8U && domain.is_signed) {
        return "i8";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 32U && domain.is_signed) {
        return "i32";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 64U && domain.is_signed) {
        return "i64";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 8U && !domain.is_signed) {
        return "u8";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 32U && !domain.is_signed) {
        return "u32";
    }
    if (domain.kind == SIM_SCALAR_DOMAIN_INTEGER && domain.bit_width == 64U && !domain.is_signed) {
        return "u64";
    }
    return sim_scalar_domain_kind_name(domain.kind);
}

bool sim_scalar_domain_from_name(const char* text, SimScalarDomain* out_domain) {
    if (text == NULL || out_domain == NULL) {
        return false;
    }

    if (sim_field_text_iequals(text, "unknown")) {
        *out_domain = sim_scalar_domain_unknown();
        return true;
    }
    if (sim_field_text_iequals(text, "f64") || sim_field_text_iequals(text, "real")) {
        *out_domain = sim_scalar_domain_f64();
        return true;
    }
    if (sim_field_text_iequals(text, "c64") || sim_field_text_iequals(text, "complex")) {
        *out_domain = sim_scalar_domain_c64();
        return true;
    }
    if (sim_field_text_iequals(text, "i8")) {
        *out_domain = sim_scalar_domain_i8();
        return true;
    }
    if (sim_field_text_iequals(text, "i32")) {
        *out_domain = sim_scalar_domain_i32();
        return true;
    }
    if (sim_field_text_iequals(text, "i64") || sim_field_text_iequals(text, "integer")) {
        *out_domain = sim_scalar_domain_i64();
        return true;
    }
    if (sim_field_text_iequals(text, "u32")) {
        *out_domain = sim_scalar_domain_u32();
        return true;
    }
    if (sim_field_text_iequals(text, "u8")) {
        *out_domain = sim_scalar_domain_u8();
        return true;
    }
    if (sim_field_text_iequals(text, "u64")) {
        *out_domain = sim_scalar_domain_u64();
        return true;
    }
    return false;
}

bool sim_scalar_domain_equal(SimScalarDomain lhs, SimScalarDomain rhs) {
    return lhs.kind == rhs.kind && lhs.bit_width == rhs.bit_width &&
           lhs.is_signed == rhs.is_signed && lhs.modulus == rhs.modulus;
}

bool sim_scalar_domain_validate(SimScalarDomain domain) {
    switch (domain.kind) {
        case SIM_SCALAR_DOMAIN_UNKNOWN:
            return domain.bit_width == 0U && !domain.is_signed && domain.modulus == 0U;
        case SIM_SCALAR_DOMAIN_REAL:
        case SIM_SCALAR_DOMAIN_COMPLEX:
            return domain.bit_width == 64U && domain.modulus == 0U;
        case SIM_SCALAR_DOMAIN_INTEGER:
            return (domain.bit_width == 8U || domain.bit_width == 32U || domain.bit_width == 64U) &&
                   domain.modulus == 0U;
        case SIM_SCALAR_DOMAIN_MODULAR:
            return domain.bit_width > 0U && domain.bit_width <= 64U && domain.modulus > 1U;
        default:
            return false;
    }
}

bool sim_scalar_domain_is_complex(SimScalarDomain domain) {
    return domain.kind == SIM_SCALAR_DOMAIN_COMPLEX;
}

bool sim_scalar_domain_is_integer(SimScalarDomain domain) {
    return domain.kind == SIM_SCALAR_DOMAIN_INTEGER;
}

uint32_t sim_scalar_domain_capabilities(SimScalarDomain domain) {
    if (!sim_scalar_domain_validate(domain)) {
        return SIM_SCALAR_CAP_NONE;
    }

    switch (domain.kind) {
        case SIM_SCALAR_DOMAIN_REAL:
            return SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC |
                   SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC | SIM_SCALAR_CAP_DIVISION |
                   SIM_SCALAR_CAP_ORDERING | SIM_SCALAR_CAP_FLOOR | SIM_SCALAR_CAP_MODULO |
                   SIM_SCALAR_CAP_ANALYTIC_CALL | SIM_SCALAR_CAP_POWER;
        case SIM_SCALAR_DOMAIN_COMPLEX:
            return SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC |
                   SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC | SIM_SCALAR_CAP_DIVISION |
                   SIM_SCALAR_CAP_CONJUGATION | SIM_SCALAR_CAP_COMPLEX_ROTATION |
                   SIM_SCALAR_CAP_POWER;
        case SIM_SCALAR_DOMAIN_INTEGER:
            return SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC |
                   SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC | SIM_SCALAR_CAP_DIVISION |
                   SIM_SCALAR_CAP_ORDERING | SIM_SCALAR_CAP_FLOOR | SIM_SCALAR_CAP_MODULO |
                   SIM_SCALAR_CAP_POWER;
        case SIM_SCALAR_DOMAIN_MODULAR:
            return SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC | SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC;
        case SIM_SCALAR_DOMAIN_UNKNOWN:
        default:
            return SIM_SCALAR_CAP_NONE;
    }
}

bool sim_scalar_domain_supports(SimScalarDomain domain, uint32_t capability_mask) {
    uint32_t available = sim_scalar_domain_capabilities(domain);
    if (capability_mask == 0U) {
        return true;
    }
    return (available & capability_mask) == capability_mask;
}

const char* sim_field_domain_name(SimFieldDomain domain) {
    switch (domain) {
        case SIM_FIELD_DOMAIN_PHYSICAL:
            return "physical";
        case SIM_FIELD_DOMAIN_SPECTRAL:
            return "spectral";
        case SIM_FIELD_DOMAIN_HYBRID:
            return "hybrid";
        case SIM_FIELD_DOMAIN_UNKNOWN:
        default:
            return "unknown";
    }
}

const char* sim_field_value_kind_name(SimFieldValueKind kind) {
    switch (kind) {
        case SIM_FIELD_VALUE_REAL_SCALAR:
            return "real";
        case SIM_FIELD_VALUE_COMPLEX_SCALAR:
            return "complex";
        case SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT:
            return "complex(imag-zero-constraint)";
        case SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT:
            return "complex(real-constraint)";
        case SIM_FIELD_VALUE_UNKNOWN:
            return "unknown";
        default:
            return "unknown";
    }
}

bool sim_field_domain_from_name(const char* text, SimFieldDomain* out_domain) {
    if (text == NULL || out_domain == NULL) {
        return false;
    }

    if (sim_field_text_iequals(text, "unknown")) {
        *out_domain = SIM_FIELD_DOMAIN_UNKNOWN;
        return true;
    }
    if (sim_field_text_iequals(text, "physical") || sim_field_text_iequals(text, "spatial")) {
        *out_domain = SIM_FIELD_DOMAIN_PHYSICAL;
        return true;
    }
    if (sim_field_text_iequals(text, "spectral") || sim_field_text_iequals(text, "frequency")) {
        *out_domain = SIM_FIELD_DOMAIN_SPECTRAL;
        return true;
    }
    if (sim_field_text_iequals(text, "hybrid")) {
        *out_domain = SIM_FIELD_DOMAIN_HYBRID;
        return true;
    }
    return false;
}

bool sim_field_value_kind_from_name(const char* text, SimFieldValueKind* out_kind) {
    if (text == NULL || out_kind == NULL) {
        return false;
    }

    if (sim_field_text_iequals(text, "unknown")) {
        *out_kind = SIM_FIELD_VALUE_UNKNOWN;
        return true;
    }
    if (sim_field_text_iequals(text, "real") || sim_field_text_iequals(text, "real_scalar") ||
        sim_field_text_iequals(text, "scalar")) {
        *out_kind = SIM_FIELD_VALUE_REAL_SCALAR;
        return true;
    }
    if (sim_field_text_iequals(text, "complex") || sim_field_text_iequals(text, "complex_scalar")) {
        *out_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        return true;
    }
    if (sim_field_text_iequals(text, "complex_imag_zero_constraint") ||
        sim_field_text_iequals(text, "imag_zero_constraint") ||
        sim_field_text_iequals(text, "imag_zero")) {
        *out_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;
        return true;
    }
    if (sim_field_text_iequals(text, "complex_real_constraint") ||
        sim_field_text_iequals(text, "real_constraint") ||
        sim_field_text_iequals(text, "hermitian") ||
        sim_field_text_iequals(text, "conjugate_symmetric")) {
        *out_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;
        return true;
    }
    return false;
}

SimResult sim_field_validate_representation(const SimField* field, SimFieldRepresentation repr) {
    bool storage_complex;
    bool repr_requires_complex;

    if (!sim_field_is_initialized(field)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    switch (repr.domain) {
        case SIM_FIELD_DOMAIN_PHYSICAL:
        case SIM_FIELD_DOMAIN_SPECTRAL:
            break;
        case SIM_FIELD_DOMAIN_HYBRID:
            return SIM_RESULT_NOT_SUPPORTED;
        case SIM_FIELD_DOMAIN_UNKNOWN:
        default:
            return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (repr.value_kind == SIM_FIELD_VALUE_UNKNOWN) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_value_kind_has_spectral_real_constraint(repr.value_kind) &&
        repr.domain != SIM_FIELD_DOMAIN_SPECTRAL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL &&
        repr.value_kind == SIM_FIELD_VALUE_REAL_SCALAR) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    repr_requires_complex = sim_field_representation_requires_complex_storage(repr);
    if (sim_scalar_domain_validate(field->scalar_domain) &&
        field->scalar_domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN &&
        sim_scalar_domain_is_complex(field->scalar_domain) != repr_requires_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    storage_complex      = sim_field_storage_is_complex(field);
    if (repr_requires_complex != storage_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    return SIM_RESULT_OK;
}

SimResult sim_field_set_representation(SimField* field, SimFieldRepresentation repr) {
    SimResult rc;
    if (!sim_field_is_initialized(field)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rc = sim_field_validate_representation(field, repr);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    field->repr         = repr;
    field->complex_mode = sim_field_representation_requires_complex_storage(repr);
    return SIM_RESULT_OK;
}

SimResult sim_field_set_scalar_domain(SimField* field, SimScalarDomain domain) {
    bool repr_requires_complex;

    if (!sim_field_is_initialized(field) || !sim_scalar_domain_validate(domain)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN &&
        !sim_field_storage_matches_scalar_domain(field->element_size, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    repr_requires_complex = sim_field_representation_requires_complex_storage(
        sim_field_representation(field));
    if (domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN &&
        sim_scalar_domain_is_complex(domain) != repr_requires_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    field->scalar_domain = domain;
    return SIM_RESULT_OK;
}
