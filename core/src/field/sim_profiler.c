/**
 * @file sim_profiler.c
 * @brief Runtime profiler implementation.
 */

#include "oakfield/sim_profiler.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static uint64_t sim_profiler_now_ns(void)
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

SimResult sim_profiler_init(SimProfiler *profiler, size_t operators)
{
    if (profiler == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void)memset(profiler, 0, sizeof(*profiler));

    if (operators == 0U)
    {
        return SIM_RESULT_OK;
    }

    profiler->counters = (SimProfilerCounter *)calloc(operators, sizeof(SimProfilerCounter));
    if (profiler->counters == NULL)
    {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    profiler->counter_count = operators;
    profiler->frame_active = false;
    profiler->frame_start_ns = 0U;
    profiler->frame_end_ns = 0U;
    return SIM_RESULT_OK;
}

void sim_profiler_destroy(SimProfiler *profiler)
{
    if (profiler == NULL)
    {
        return;
    }

    free(profiler->counters);
    profiler->counters = NULL;
    profiler->counter_count = 0U;
    profiler->frame_active = false;
    profiler->frame_start_ns = 0U;
    profiler->frame_end_ns = 0U;
}

void sim_profiler_begin_frame(SimProfiler *profiler)
{
    if (profiler == NULL)
    {
        return;
    }

    if (profiler->counters != NULL && profiler->counter_count > 0U)
    {
        (void)memset(profiler->counters, 0, profiler->counter_count * sizeof(SimProfilerCounter));
    }

    profiler->frame_start_ns = sim_profiler_now_ns();
    profiler->frame_active = true;
}

void sim_profiler_end_frame(SimProfiler *profiler)
{
    if (profiler == NULL)
    {
        return;
    }

    if (profiler->frame_active)
    {
        profiler->frame_end_ns = sim_profiler_now_ns();
        profiler->frame_active = false;
    }
}

void sim_profiler_record_operator(SimProfiler *profiler,
                                  size_t operator_index,
                                  uint64_t duration_ns)
{
    SimProfilerCounter *counter;

    if (profiler == NULL || operator_index >= profiler->counter_count)
    {
        return;
    }

    counter = &profiler->counters[operator_index];
    counter->inclusive_ns += duration_ns;
    counter->invocations += 1U;
}

void sim_profiler_record_operator_delta(SimProfiler *profiler,
                                        size_t operator_index,
                                        double delta_rms,
                                        uint64_t sample_count)
{
    if (profiler == NULL || profiler->counters == NULL || operator_index >= profiler->counter_count)
        return;
    SimProfilerCounter *counter = &profiler->counters[operator_index];
    /* Aggregate sum of squared RMS (so we can later average if needed). */
    __atomic_fetch_add((uint64_t *)&counter->delta_sample_count, (uint64_t)sample_count, __ATOMIC_RELAXED);
    /* Accumulate a scaled value: we store sum(delta_rms * sample_count) as double via memcpy to a uint64_t storage for atomic add. */
    double add_val = delta_rms * (double)sample_count;
    uint64_t add_bits;
    memcpy(&add_bits, &add_val, sizeof(add_bits));
    /* Non-atomic addition of double via lockless approach: not portable — fall back to atomic via CAS loop on 64-bit integer. */
    uint64_t old_bits = __atomic_load_n((uint64_t *)&counter->delta_rms_sum, __ATOMIC_RELAXED);
    uint64_t new_bits;
    double old_val, new_val;
    do
    {
        memcpy(&old_val, &old_bits, sizeof(old_val));
        new_val = old_val + add_val;
        memcpy(&new_bits, &new_val, sizeof(new_bits));
    } while (!__atomic_compare_exchange_n((uint64_t *)&counter->delta_rms_sum, &old_bits, new_bits, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

SimResult sim_profiler_snapshot(const SimProfiler *profiler,
                                SimProfilerSnapshot *out_snapshot)
{
    double total_invocations = 0.0;
    double total_time = 0.0;
    size_t i;

    if (profiler == NULL || out_snapshot == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (profiler->frame_start_ns == 0U && profiler->frame_end_ns == 0U)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    out_snapshot->frame_start_ns = profiler->frame_start_ns;
    out_snapshot->frame_end_ns = profiler->frame_end_ns;
    out_snapshot->total_ns = (profiler->frame_end_ns >= profiler->frame_start_ns)
                                 ? (profiler->frame_end_ns - profiler->frame_start_ns)
                                 : 0U;

    for (i = 0; i < profiler->counter_count; ++i)
    {
        total_time += (double)profiler->counters[i].inclusive_ns;
        total_invocations += (double)profiler->counters[i].invocations;
    }

    if (total_invocations > 0.0)
    {
        out_snapshot->average_operator_ns = total_time / total_invocations;
    }
    else
    {
        out_snapshot->average_operator_ns = 0.0;
    }

    return SIM_RESULT_OK;
}
