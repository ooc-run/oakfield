/**
 * @file async_logger.c
 * @brief Lock-free asynchronous logger implementation.
 */

#include "oakfield/async_logger.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#define SIM_ATOMIC_FETCH_ADD(ptr, value) __sync_fetch_and_add((size_t *)(ptr), (size_t)(value))
#define SIM_ATOMIC_LOAD(ptr) __sync_fetch_and_add((size_t *)(ptr), (size_t)0)
#define SIM_ATOMIC_STORE(ptr, value) (void)__sync_lock_test_and_set((size_t *)(ptr), (size_t)(value))
#define SIM_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired) __sync_bool_compare_and_swap((size_t *)(ptr), (expected), (desired))
#define SIM_ATOMIC_FETCH_SUB(ptr, value) __sync_fetch_and_sub((size_t *)(ptr), (size_t)(value))

static size_t sim_async_logger_next_pow2(size_t value)
{
    size_t n = 1U;

    if (value == 0U)
    {
        return 64U;
    }

    while (n < value)
    {
        n <<= 1U;
    }

    if (n < 64U)
    {
        n = 64U;
    }

    return n;
}

static uint64_t sim_async_logger_now_ns(void)
{
#if defined(__APPLE__)
    static mach_timebase_info_data_t info = {0, 0};
    uint64_t ticks = mach_absolute_time();
    if (info.denom == 0U)
    {
        mach_timebase_info(&info);
    }
    return ticks * (uint64_t)info.numer / (uint64_t)info.denom;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        return 0U;
    }
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
#endif
}

static uint32_t sim_async_logger_thread_id(void)
{
    union
    {
        pthread_t thread;
        uintptr_t value;
    } u;

    u.thread = pthread_self();
    return (uint32_t)(u.value & 0xFFFFFFFFU);
}

SimResult sim_async_logger_init(SimAsyncLogger *logger, size_t capacity)
{
    if (logger == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void)memset(logger, 0, sizeof(*logger));

    logger->capacity = sim_async_logger_next_pow2(capacity);
    logger->records = (SimAsyncLogRecord *)calloc(logger->capacity, sizeof(SimAsyncLogRecord));
    if (logger->records == NULL)
    {
        (void)memset(logger, 0, sizeof(*logger));
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    logger->head = 0U;
    logger->tail = 0U;
    return SIM_RESULT_OK;
}

void sim_async_logger_destroy(SimAsyncLogger *logger)
{
    if (logger == NULL)
    {
        return;
    }

    free(logger->records);
    logger->records = NULL;
    logger->capacity = 0U;
    logger->head = 0U;
    logger->tail = 0U;
}

SimResult sim_async_logger_log(SimAsyncLogger *logger,
                               SimLogLevel level,
                               const char *message)
{
    size_t slot;
    size_t index;
    size_t tail;
    SimAsyncLogRecord *record;

    if (logger == NULL || logger->records == NULL || message == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    slot = SIM_ATOMIC_FETCH_ADD(&logger->head, 1U);
    index = slot & (logger->capacity - 1U);

    tail = SIM_ATOMIC_LOAD(&logger->tail);
    if ((slot - tail) >= logger->capacity)
    {
        (void)SIM_ATOMIC_COMPARE_EXCHANGE(&logger->tail, tail, tail + 1U);
    }

    record = &logger->records[index];
    record->timestamp_ns = sim_async_logger_now_ns();
    record->thread_id = sim_async_logger_thread_id();
    record->level = level;

    (void)snprintf(record->message, sizeof(record->message), "%s", message);
    record->message[sizeof(record->message) - 1U] = '\0';

    return SIM_RESULT_OK;
}

bool sim_async_logger_pop(SimAsyncLogger *logger, SimAsyncLogRecord *out_record)
{
    size_t tail;
    size_t head;
    size_t index;
    SimAsyncLogRecord *src;

    if (logger == NULL || logger->records == NULL || out_record == NULL)
    {
        return false;
    }

    for (;;)
    {
        tail = SIM_ATOMIC_LOAD(&logger->tail);
        head = SIM_ATOMIC_LOAD(&logger->head);

        if (tail >= head)
        {
            return false;
        }

        if (SIM_ATOMIC_COMPARE_EXCHANGE(&logger->tail, tail, tail + 1U))
        {
            break;
        }
    }

    index = tail & (logger->capacity - 1U);
    src = &logger->records[index];
    *out_record = *src;
    return true;
}

void sim_async_logger_clear(SimAsyncLogger *logger)
{
    size_t head;

    if (logger == NULL)
    {
        return;
    }

    head = SIM_ATOMIC_LOAD(&logger->head);
    SIM_ATOMIC_STORE(&logger->tail, head);
}
