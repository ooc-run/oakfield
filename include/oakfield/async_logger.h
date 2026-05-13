/**
 * @file async_logger.h
 * @brief Lock-free asynchronous logging facilities for libsimcore runtime.
 */
#ifndef OAKFIELD_ASYNC_LOGGER_H
#define OAKFIELD_ASYNC_LOGGER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum length, in bytes, of a log message (excluding the null terminator). */
#define SIM_ASYNC_LOGGER_MESSAGE_MAX 255U

/**
 * @brief Severity levels emitted by the asynchronous logger.
 */
typedef enum SimLogLevel {
    SIM_LOG_LEVEL_TRACE = 0, /**< Verbose diagnostic information. */
    SIM_LOG_LEVEL_DEBUG,     /**< Debug-level information. */
    SIM_LOG_LEVEL_INFO,      /**< Informational message. */
    SIM_LOG_LEVEL_WARN,      /**< Warning that does not halt execution. */
    SIM_LOG_LEVEL_ERROR,     /**< Recoverable error. */
    SIM_LOG_LEVEL_FATAL      /**< Fatal error after which the system cannot continue. */
} SimLogLevel;

/**
 * @brief Log entry materialized when records are drained from the ring buffer.
 */
typedef struct SimAsyncLogRecord {
    uint64_t timestamp_ns;                           /**< Monotonic timestamp in nanoseconds. */
    uint32_t thread_id;                              /**< Lightweight thread identifier. */
    SimLogLevel level;                               /**< Severity level. */
    char message[SIM_ASYNC_LOGGER_MESSAGE_MAX + 1U]; /**< Null-terminated payload. */
} SimAsyncLogRecord;

/**
 * @brief Lock-free ring buffer logger supporting multiple producers and a single consumer.
 */
typedef struct SimAsyncLogger {
    SimAsyncLogRecord *records; /**< Ring buffer storage. */
    size_t capacity;            /**< Power-of-two capacity for @ref records. */
    volatile size_t head;       /**< Producer write index (monotonic). */
    volatile size_t tail;       /**< Consumer read index (monotonic). */
} SimAsyncLogger;

/**
 * @brief Initialize the asynchronous logger with a ring buffer of at least @p capacity elements.
 *
 * @param[out] logger Logger instance to initialize.
 * @param capacity Requested capacity; rounded up to the next power of two (minimum 64).
 * @return #SIM_RESULT_OK on success or an error code on failure.
 */
SimResult sim_async_logger_init(SimAsyncLogger *logger, size_t capacity);

/**
 * @brief Destroy an asynchronous logger instance and release its storage.
 *
 * @param logger Logger to destroy; may be NULL.
 */
void sim_async_logger_destroy(SimAsyncLogger *logger);

/**
 * @brief Attempt to append a log record without blocking.
 *
 * When the ring buffer is full older messages are overwritten.
 *
 * @param logger Logger instance.
 * @param level Message severity.
 * @param message Null-terminated string to append.
 * @return #SIM_RESULT_OK on success or an error code if the record could not be written.
 */
SimResult sim_async_logger_log(SimAsyncLogger *logger, SimLogLevel level, const char *message);

/**
 * @brief Drain the next pending log record.
 *
 * @param logger Logger instance.
 * @param[out] out_record Destination for the oldest pending record.
 * @return true when a record was produced, false if no messages were available.
 */
bool sim_async_logger_pop(SimAsyncLogger *logger, SimAsyncLogRecord *out_record);

/**
 * @brief Clear all pending log entries without releasing internal storage.
 *
 * @param logger Logger instance.
 */
void sim_async_logger_clear(SimAsyncLogger *logger);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_ASYNC_LOGGER_H */
