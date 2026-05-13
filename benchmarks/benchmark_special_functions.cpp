#include <oakfield/math.h>
#include <oakfield/math/airy.h>
#include <oakfield/math/bessel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef OAKFIELD_BENCH_HAVE_BOOST_MATH
#define OAKFIELD_BENCH_HAVE_BOOST_MATH 0
#endif

#ifndef OAKFIELD_BENCH_HAVE_GSL
#define OAKFIELD_BENCH_HAVE_GSL 0
#endif

#ifndef OAKFIELD_BENCH_HAVE_MPFR
#define OAKFIELD_BENCH_HAVE_MPFR 0
#endif

#ifndef OAKFIELD_ENABLE_ZETA_CORE
#define OAKFIELD_ENABLE_ZETA_CORE 0
#endif

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
#include <boost/math/special_functions/airy.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/zeta.hpp>
#endif

#if OAKFIELD_BENCH_HAVE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_zeta.h>
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
#include <mpfr.h>
#endif

namespace {

volatile double g_sink = 0.0;

enum class ColorMode {
    auto_detect,
    always,
    never,
};

struct Options {
    int       iterations          = 128;
    int       samples             = 256;
    int       zeta_iterations     = -1;
    int       mpfr_iterations     = -1;
    int       mpfr_precision_bits = 128;
    double    adaptive_tolerance  = 1.0e-16;
    bool      include_zeta        = true;
    bool      key_value_output    = false;
    ColorMode color_mode          = ColorMode::auto_detect;
};

struct DeltaStats {
    double max_abs = std::numeric_limits<double>::quiet_NaN();
    double rms     = std::numeric_limits<double>::quiet_NaN();
    int    samples = 0;
};

struct BenchmarkRow {
    const char* function     = "";
    const char* library_name = "";
    double      seconds      = 0.0;
    double      calls        = 0.0;
    double      checksum     = 0.0;
    DeltaStats  delta;
    bool        has_delta   = false;
    bool        skipped     = false;
    bool        is_oracle   = false;
    const char* skip_reason = "";
};

struct ColorPalette {
    bool        enabled  = false;
    const char* fastest  = "";
    const char* accurate = "";
    const char* reset    = "";
};

enum class ParseStatus {
    no_match,
    ok,
    invalid,
};

static ParseStatus
parse_int_arg(const char* arg, const char* prefix, int min_value, int max_value, int* out_value) {
    const std::size_t prefix_len = std::strlen(prefix);
    char*             endptr     = nullptr;
    long              parsed     = 0;

    if (std::strncmp(arg, prefix, prefix_len) != 0) {
        return ParseStatus::no_match;
    }

    parsed = std::strtol(arg + prefix_len, &endptr, 10);
    if (endptr == arg + prefix_len || *endptr != '\0' || parsed < min_value || parsed > max_value) {
        return ParseStatus::invalid;
    }

    *out_value = static_cast<int>(parsed);
    return ParseStatus::ok;
}

static ParseStatus parse_double_arg(const char* arg,
                                    const char* prefix,
                                    double      min_value,
                                    double      max_value,
                                    double*     out_value) {
    const std::size_t prefix_len = std::strlen(prefix);
    char*             endptr     = nullptr;
    double            parsed     = 0.0;

    if (std::strncmp(arg, prefix, prefix_len) != 0) {
        return ParseStatus::no_match;
    }

    parsed = std::strtod(arg + prefix_len, &endptr);
    if (endptr == arg + prefix_len || *endptr != '\0' || !std::isfinite(parsed) ||
        parsed < min_value || parsed > max_value) {
        return ParseStatus::invalid;
    }

    *out_value = parsed;
    return ParseStatus::ok;
}

static void print_usage(void) {
    std::printf("usage: benchmark_special_functions "
                "[--iterations=N] [--samples=N] [--zeta-iterations=N] "
                "[--mpfr-iterations=N] [--mpfr-precision=BITS] "
                "[--adaptive-tol=EPS] [--format=table|kv] "
                "[--color=auto|always|never] [--no-zeta]\n");
}

static int parse_options(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        ParseStatus status = ParseStatus::no_match;

        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 1;
        }
        if (std::strcmp(argv[i], "--no-zeta") == 0) {
            options->include_zeta = false;
            continue;
        }
        if (std::strcmp(argv[i], "--format=table") == 0) {
            options->key_value_output = false;
            continue;
        }
        if (std::strcmp(argv[i], "--format=kv") == 0) {
            options->key_value_output = true;
            continue;
        }
        if (std::strcmp(argv[i], "--color=auto") == 0) {
            options->color_mode = ColorMode::auto_detect;
            continue;
        }
        if (std::strcmp(argv[i], "--color=always") == 0) {
            options->color_mode = ColorMode::always;
            continue;
        }
        if (std::strcmp(argv[i], "--color=never") == 0) {
            options->color_mode = ColorMode::never;
            continue;
        }

        status = parse_int_arg(argv[i], "--iterations=", 1, 1000000, &options->iterations);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid iteration count: %s\n", argv[i]);
            return -1;
        }

        status = parse_int_arg(argv[i], "--samples=", 1, 1000000, &options->samples);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid sample count: %s\n", argv[i]);
            return -1;
        }

        status =
            parse_int_arg(argv[i], "--zeta-iterations=", 1, 1000000, &options->zeta_iterations);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid zeta iteration count: %s\n", argv[i]);
            return -1;
        }

        status =
            parse_int_arg(argv[i], "--mpfr-iterations=", 1, 1000000, &options->mpfr_iterations);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid MPFR iteration count: %s\n", argv[i]);
            return -1;
        }

        status =
            parse_int_arg(argv[i], "--mpfr-precision=", 24, 4096, &options->mpfr_precision_bits);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid MPFR precision: %s\n", argv[i]);
            return -1;
        }

        status =
            parse_double_arg(argv[i], "--adaptive-tol=", 0.0, 1.0, &options->adaptive_tolerance);
        if (status == ParseStatus::ok) {
            continue;
        }
        if (status == ParseStatus::invalid) {
            std::fprintf(stderr, "invalid adaptive tolerance: %s\n", argv[i]);
            return -1;
        }

        std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
        return -1;
    }

    if (options->zeta_iterations < 0) {
        options->zeta_iterations = std::max(1, options->iterations / 16);
    }
    if (options->mpfr_iterations < 0) {
        options->mpfr_iterations = std::max(1, options->iterations / 32);
    }

    return 0;
}

static double seconds_now(void) {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static bool stdout_is_tty(void) {
#if defined(_WIN32)
    return false;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

static bool should_use_color(const Options& options) {
    if (options.key_value_output || options.color_mode == ColorMode::never) {
        return false;
    }
    if (options.color_mode == ColorMode::always) {
        return true;
    }
    const char* no_color = std::getenv("NO_COLOR");
    const char* term     = std::getenv("TERM");
    return no_color == nullptr && stdout_is_tty() &&
           (term == nullptr || std::strcmp(term, "dumb") != 0);
}

static ColorPalette make_colors(const Options& options) {
    ColorPalette colors;
    colors.enabled = should_use_color(options);
    if (colors.enabled) {
        colors.fastest  = "\033[1;32m";
        colors.accurate = "\033[1;36m";
        colors.reset    = "\033[0m";
    }
    return colors;
}

static std::vector<double> make_positive_inputs(int samples) {
    std::vector<double> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t      = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        const double ripple = 0.03125 * std::sin(47.0 * t);
        inputs.push_back(0.125 + 31.875 * t + ripple);
    }

    return inputs;
}

static std::vector<double> make_airy_inputs(int samples) {
    std::vector<double> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        inputs.push_back(-12.0 + 20.0 * t);
    }

    return inputs;
}

static std::vector<double> make_bessel_inputs(int samples) {
    std::vector<double> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        inputs.push_back(45.0 * t);
    }

    return inputs;
}

static std::vector<double> make_zeta_inputs(int samples) {
    std::vector<double> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        inputs.push_back(1.125 + 18.875 * t);
    }

    return inputs;
}

static std::vector<SimComplexDouble> make_special_complex_inputs(int samples) {
    std::vector<SimComplexDouble> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        inputs.push_back((SimComplexDouble) { 0.25 + 19.75 * t, -3.0 + 6.0 * t });
    }

    return inputs;
}

static std::vector<SimComplexDouble> make_zeta_complex_inputs(int samples) {
    std::vector<SimComplexDouble> inputs;
    inputs.reserve(static_cast<std::size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double t     = (static_cast<double>(i) + 0.5) / static_cast<double>(samples);
        const double sigma = (i % 2 == 0) ? 0.5 : 2.0;
        inputs.push_back((SimComplexDouble) { sigma, 1.0 + 19.0 * t });
    }

    return inputs;
}

template <typename Func>
static std::vector<double> evaluate_unary(const std::vector<double>& inputs, Func func) {
    std::vector<double> values;
    values.reserve(inputs.size());

    for (double x : inputs) {
        values.push_back(static_cast<double>(func(x)));
    }

    return values;
}

template <typename Func>
static std::vector<double> evaluate_indexed(const std::vector<double>& inputs, Func func) {
    std::vector<double> values;
    values.reserve(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        values.push_back(static_cast<double>(func(i, inputs[i])));
    }

    return values;
}

template <typename Func>
static std::vector<SimComplexDouble>
evaluate_complex_unary(const std::vector<SimComplexDouble>& inputs, Func func) {
    std::vector<SimComplexDouble> values;
    values.reserve(inputs.size());

    for (SimComplexDouble z : inputs) {
        values.push_back(func(z));
    }

    return values;
}

template <typename Func>
static DeltaStats compute_delta_unary(const std::vector<double>& inputs,
                                      const std::vector<double>& reference,
                                      Func                       func) {
    DeltaStats  stats;
    long double sum_sq = 0.0L;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const double value    = static_cast<double>(func(inputs[i]));
        const double expected = reference[i];
        if (std::isfinite(value) && std::isfinite(expected)) {
            const double delta = std::fabs(value - expected);
            if (stats.samples == 0 || delta > stats.max_abs) {
                stats.max_abs = delta;
            }
            sum_sq += static_cast<long double>(delta) * static_cast<long double>(delta);
            ++stats.samples;
        }
    }

    if (stats.samples > 0) {
        stats.rms =
            std::sqrt(static_cast<double>(sum_sq / static_cast<long double>(stats.samples)));
    }

    return stats;
}

template <typename Func>
static DeltaStats compute_delta_complex_unary(const std::vector<SimComplexDouble>& inputs,
                                              const std::vector<SimComplexDouble>& reference,
                                              Func                                 func) {
    DeltaStats  stats;
    long double sum_sq = 0.0L;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const SimComplexDouble value    = func(inputs[i]);
        const SimComplexDouble expected = reference[i];
        if (std::isfinite(value.re) && std::isfinite(value.im) && std::isfinite(expected.re) &&
            std::isfinite(expected.im)) {
            const double delta_re = value.re - expected.re;
            const double delta_im = value.im - expected.im;
            const double delta    = std::hypot(delta_re, delta_im);
            if (stats.samples == 0 || delta > stats.max_abs) {
                stats.max_abs = delta;
            }
            sum_sq += static_cast<long double>(delta) * static_cast<long double>(delta);
            ++stats.samples;
        }
    }

    if (stats.samples > 0) {
        stats.rms =
            std::sqrt(static_cast<double>(sum_sq / static_cast<long double>(stats.samples)));
    }

    return stats;
}

template <typename Func>
static DeltaStats compute_delta_indexed(const std::vector<double>& inputs,
                                        const std::vector<double>& reference,
                                        Func                       func) {
    DeltaStats  stats;
    long double sum_sq = 0.0L;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const double value    = static_cast<double>(func(i, inputs[i]));
        const double expected = reference[i];
        if (std::isfinite(value) && std::isfinite(expected)) {
            const double delta = std::fabs(value - expected);
            if (stats.samples == 0 || delta > stats.max_abs) {
                stats.max_abs = delta;
            }
            sum_sq += static_cast<long double>(delta) * static_cast<long double>(delta);
            ++stats.samples;
        }
    }

    if (stats.samples > 0) {
        stats.rms =
            std::sqrt(static_cast<double>(sum_sq / static_cast<long double>(stats.samples)));
    }

    return stats;
}

static void print_table_header(const char* function, const char* reference_label) {
    std::printf("\n%s (reference: %s)\n", function, reference_label);
    std::printf("  %-22s %10s %10s %14s %16s %14s %14s  %s\n",
                "library",
                "seconds",
                "calls",
                "calls/sec",
                "checksum",
                "max_abs_delta",
                "rms_delta",
                "markers");
    std::printf("  %-22s %10s %10s %14s %16s %14s %14s  %s\n",
                "----------------------",
                "----------",
                "----------",
                "--------------",
                "----------------",
                "--------------",
                "--------------",
                "-------");
}

static double row_rate(const BenchmarkRow& row) {
    return row.seconds > 0.0 ? row.calls / row.seconds : 0.0;
}

static bool row_has_accuracy(const BenchmarkRow& row) {
    return row.has_delta && row.delta.samples > 0 && std::isfinite(row.delta.rms) &&
           std::isfinite(row.delta.max_abs);
}

static const char* reference_name(void) {
#if OAKFIELD_BENCH_HAVE_MPFR
    return "mpfr_oracle";
#else
    return "oakfield_reference";
#endif
}

static BenchmarkRow
make_skipped_row(const char* function, const char* library_name, const char* reason) {
    BenchmarkRow row;
    row.function     = function;
    row.library_name = library_name;
    row.skipped      = true;
    row.skip_reason  = reason;
    return row;
}

static BenchmarkRow mark_oracle(BenchmarkRow row) {
    row.is_oracle = true;
    return row;
}

static void print_result_key_value(const BenchmarkRow& row,
                                   bool                is_fastest,
                                   bool                is_most_accurate,
                                   const char*         reference_label) {
    if (row.skipped) {
        std::printf("SPECIAL function=%s library=%s skipped=1 reason=%s\n",
                    row.function,
                    row.library_name,
                    row.skip_reason);
        return;
    }

    std::printf("SPECIAL function=%s library=%s seconds=%.6f calls=%.0f "
                "calls_per_sec=%.3f checksum=%.12g",
                row.function,
                row.library_name,
                row.seconds,
                row.calls,
                row_rate(row),
                row.checksum);
    if (row_has_accuracy(row)) {
        std::printf(" max_abs_delta=%.12g rms_delta=%.12g delta_samples=%d "
                    "reference=%s",
                    row.delta.max_abs,
                    row.delta.rms,
                    row.delta.samples,
                    reference_label);
    }
    if (is_fastest) {
        std::printf(" fastest=1");
    }
    if (is_most_accurate) {
        std::printf(" most_accurate=1");
    }
    if (row.is_oracle) {
        std::printf(" oracle=1");
    }
    std::printf("\n");
}

static void
print_markers(const ColorPalette& colors, bool is_fastest, bool is_most_accurate, bool is_oracle) {
    if (!is_fastest && !is_most_accurate && !is_oracle) {
        return;
    }

    bool needs_comma = false;
    if (is_fastest) {
        std::printf("%sfastest%s", colors.fastest, colors.reset);
        needs_comma = true;
    }
    if (is_most_accurate) {
        std::printf("%s%saccurate%s", needs_comma ? "," : "", colors.accurate, colors.reset);
        needs_comma = true;
    }
    if (is_oracle) {
        std::printf("%soracle", needs_comma ? "," : "");
    }
}

static void print_result_table(const BenchmarkRow& row,
                               const ColorPalette& colors,
                               bool                is_fastest,
                               bool                is_most_accurate) {
    if (row.skipped) {
        std::printf("  %-22s %10s %10s %14s %16s %14s %14s  %s\n",
                    row.library_name,
                    "skipped",
                    "-",
                    "-",
                    row.skip_reason,
                    "-",
                    "-",
                    "");
        return;
    }

    std::printf("  %-22s %10.6f %10.0f ", row.library_name, row.seconds, row.calls);
    if (is_fastest) {
        std::printf("%s%14.3e%s", colors.fastest, row_rate(row), colors.reset);
    } else {
        std::printf("%14.3e", row_rate(row));
    }
    std::printf(" %16.8g", row.checksum);
    if (row_has_accuracy(row)) {
        if (is_most_accurate) {
            std::printf(" %s%14.3e %14.3e%s",
                        colors.accurate,
                        row.delta.max_abs,
                        row.delta.rms,
                        colors.reset);
        } else {
            std::printf(" %14.3e %14.3e", row.delta.max_abs, row.delta.rms);
        }
    } else {
        std::printf(" %14s %14s", "-", "-");
    }
    std::printf("  ");
    print_markers(colors, is_fastest, is_most_accurate, row.is_oracle);
    std::printf("\n");
}

static void print_group_results(const Options&                   options,
                                const char*                      function,
                                const std::vector<BenchmarkRow>& rows,
                                const char*                      reference_label) {
    std::size_t fastest_index       = rows.size();
    std::size_t most_accurate_index = rows.size();
    double      best_rate           = -1.0;
    double      best_rms            = std::numeric_limits<double>::infinity();
    double      best_max_abs        = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const BenchmarkRow& row = rows[i];
        if (!row.skipped) {
            const double rate = row_rate(row);
            if (std::isfinite(rate) && rate > best_rate) {
                best_rate     = rate;
                fastest_index = i;
            }
        }
        if (row_has_accuracy(row) && !row.is_oracle) {
            if (row.delta.rms < best_rms ||
                (row.delta.rms == best_rms && row.delta.max_abs < best_max_abs)) {
                best_rms            = row.delta.rms;
                best_max_abs        = row.delta.max_abs;
                most_accurate_index = i;
            }
        }
    }

    if (most_accurate_index == rows.size()) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const BenchmarkRow& row = rows[i];
            if (row_has_accuracy(row)) {
                if (row.delta.rms < best_rms ||
                    (row.delta.rms == best_rms && row.delta.max_abs < best_max_abs)) {
                    best_rms            = row.delta.rms;
                    best_max_abs        = row.delta.max_abs;
                    most_accurate_index = i;
                }
            }
        }
    }

    if (options.key_value_output) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            print_result_key_value(
                rows[i], i == fastest_index, i == most_accurate_index, reference_label);
        }
        return;
    }

    const ColorPalette colors = make_colors(options);
    print_table_header(function, reference_label);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        print_result_table(rows[i], colors, i == fastest_index, i == most_accurate_index);
    }
}

template <typename Func>
static BenchmarkRow run_unary(const char*                function,
                              const char*                library_name,
                              const std::vector<double>& inputs,
                              int                        iterations,
                              const std::vector<double>* reference,
                              Func                       func) {
    const double start    = seconds_now();
    double       checksum = 0.0;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (double x : inputs) {
            checksum += static_cast<double>(func(x));
        }
    }

    BenchmarkRow row;
    row.function     = function;
    row.library_name = library_name;
    row.seconds      = seconds_now() - start;
    row.calls        = static_cast<double>(iterations) * static_cast<double>(inputs.size());
    row.checksum     = checksum;

    g_sink += checksum;
    if (reference != nullptr) {
        row.delta     = compute_delta_unary(inputs, *reference, func);
        row.has_delta = row.delta.samples > 0;
    }
    return row;
}

template <typename Func>
static BenchmarkRow run_complex_unary(const char*                          function,
                                      const char*                          library_name,
                                      const std::vector<SimComplexDouble>& inputs,
                                      int                                  iterations,
                                      const std::vector<SimComplexDouble>* reference,
                                      Func                                 func) {
    const double start    = seconds_now();
    double       checksum = 0.0;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (SimComplexDouble z : inputs) {
            const SimComplexDouble value = func(z);
            checksum += value.re + value.im;
        }
    }

    BenchmarkRow row;
    row.function     = function;
    row.library_name = library_name;
    row.seconds      = seconds_now() - start;
    row.calls        = static_cast<double>(iterations) * static_cast<double>(inputs.size());
    row.checksum     = checksum;

    g_sink += checksum;
    if (reference != nullptr) {
        row.delta     = compute_delta_complex_unary(inputs, *reference, func);
        row.has_delta = row.delta.samples > 0;
    }
    return row;
}

template <typename Func>
static BenchmarkRow run_indexed(const char*                function,
                                const char*                library_name,
                                const std::vector<double>& inputs,
                                int                        iterations,
                                const std::vector<double>* reference,
                                Func                       func) {
    const double start    = seconds_now();
    double       checksum = 0.0;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            checksum += static_cast<double>(func(i, inputs[i]));
        }
    }

    BenchmarkRow row;
    row.function     = function;
    row.library_name = library_name;
    row.seconds      = seconds_now() - start;
    row.calls        = static_cast<double>(iterations) * static_cast<double>(inputs.size());
    row.checksum     = checksum;

    g_sink += checksum;
    if (reference != nullptr) {
        row.delta     = compute_delta_indexed(inputs, *reference, func);
        row.has_delta = row.delta.samples > 0;
    }
    return row;
}

static int bessel_order_for_index(std::size_t index) {
    return static_cast<int>(index % 9U);
}

#if OAKFIELD_BENCH_HAVE_MPFR
class MpfrWorkspace {
  public:
    explicit MpfrWorkspace(int precision_bits) {
        mpfr_init2(input_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(output_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(center_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(plus_one_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(plus_two_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(minus_one_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(minus_two_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(step_, static_cast<mpfr_prec_t>(precision_bits));
        mpfr_init2(denominator_, static_cast<mpfr_prec_t>(precision_bits));
    }

    MpfrWorkspace(const MpfrWorkspace&)            = delete;
    MpfrWorkspace& operator=(const MpfrWorkspace&) = delete;

    ~MpfrWorkspace() {
        mpfr_clear(input_);
        mpfr_clear(output_);
        mpfr_clear(center_);
        mpfr_clear(plus_one_);
        mpfr_clear(plus_two_);
        mpfr_clear(minus_one_);
        mpfr_clear(minus_two_);
        mpfr_clear(step_);
        mpfr_clear(denominator_);
    }

    double digamma(double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_digamma(output_, input_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double trigamma(double x) {
        eval_digamma_shifted(plus_two_, x, 2.0);
        eval_digamma_shifted(plus_one_, x, 1.0);
        eval_digamma_shifted(minus_one_, x, -1.0);
        eval_digamma_shifted(minus_two_, x, -2.0);

        mpfr_sub(output_, minus_two_, plus_two_, MPFR_RNDN);
        mpfr_mul_ui(plus_one_, plus_one_, 8UL, MPFR_RNDN);
        mpfr_add(output_, output_, plus_one_, MPFR_RNDN);
        mpfr_mul_ui(minus_one_, minus_one_, 8UL, MPFR_RNDN);
        mpfr_sub(output_, output_, minus_one_, MPFR_RNDN);

        mpfr_set_d(denominator_, 12.0 * finite_difference_step(), MPFR_RNDN);
        mpfr_div(output_, output_, denominator_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double tetragamma(double x) {
        eval_digamma_shifted(plus_two_, x, 2.0);
        eval_digamma_shifted(plus_one_, x, 1.0);
        eval_digamma_shifted(center_, x, 0.0);
        eval_digamma_shifted(minus_one_, x, -1.0);
        eval_digamma_shifted(minus_two_, x, -2.0);

        mpfr_neg(output_, plus_two_, MPFR_RNDN);
        mpfr_mul_ui(plus_one_, plus_one_, 16UL, MPFR_RNDN);
        mpfr_add(output_, output_, plus_one_, MPFR_RNDN);
        mpfr_mul_ui(center_, center_, 30UL, MPFR_RNDN);
        mpfr_sub(output_, output_, center_, MPFR_RNDN);
        mpfr_mul_ui(minus_one_, minus_one_, 16UL, MPFR_RNDN);
        mpfr_add(output_, output_, minus_one_, MPFR_RNDN);
        mpfr_sub(output_, output_, minus_two_, MPFR_RNDN);

        const double step = finite_difference_step();
        mpfr_set_d(denominator_, 12.0 * step * step, MPFR_RNDN);
        mpfr_div(output_, output_, denominator_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double airy_ai(double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_ai(output_, input_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double bessel_jn(int order, double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_jn(output_, static_cast<long>(order), input_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double xi(double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_zeta(output_, input_, MPFR_RNDN);

        mpfr_set_d(plus_one_, 0.5 * x, MPFR_RNDN);
        mpfr_gamma(plus_two_, plus_one_, MPFR_RNDN);
        mpfr_mul(output_, output_, plus_two_, MPFR_RNDN);

        mpfr_const_pi(center_, MPFR_RNDN);
        mpfr_neg(minus_one_, plus_one_, MPFR_RNDN);
        mpfr_pow(minus_two_, center_, minus_one_, MPFR_RNDN);
        mpfr_mul(output_, output_, minus_two_, MPFR_RNDN);

        mpfr_set_d(denominator_, x, MPFR_RNDN);
        mpfr_set_d(step_, x - 1.0, MPFR_RNDN);
        mpfr_mul(denominator_, denominator_, step_, MPFR_RNDN);
        mpfr_div_ui(denominator_, denominator_, 2UL, MPFR_RNDN);
        mpfr_mul(output_, output_, denominator_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

    double zeta(double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_zeta(output_, input_, MPFR_RNDN);
        return mpfr_get_d(output_, MPFR_RNDN);
    }

  private:
    static double finite_difference_step(void) { return std::ldexp(1.0, -20); }

    void eval_digamma_shifted(mpfr_t out, double x, double step_scale) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        mpfr_set_d(step_, step_scale * finite_difference_step(), MPFR_RNDN);
        mpfr_add(input_, input_, step_, MPFR_RNDN);
        mpfr_digamma(out, input_, MPFR_RNDN);
    }

    mpfr_t input_;
    mpfr_t output_;
    mpfr_t center_;
    mpfr_t plus_one_;
    mpfr_t plus_two_;
    mpfr_t minus_one_;
    mpfr_t minus_two_;
    mpfr_t step_;
    mpfr_t denominator_;
};
#endif

static void run_digamma_group(const std::vector<double>& inputs, const Options& options) {
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.digamma(x); });
#else
    const std::vector<double> reference =
        evaluate_unary(inputs, [](double x) { return sim_digamma_f64_12(x); });
#endif

    rows.push_back(run_unary(
        "digamma", "oakfield_f64_12", inputs, options.iterations, &reference, [](double x) {
            return sim_digamma_f64_12(x);
        }));
    rows.push_back(run_unary(
        "digamma", "oakfield_f64_7", inputs, options.iterations, &reference, [](double x) {
            return sim_digamma_f64_7(x);
        }));
    rows.push_back(run_unary(
        "digamma", "oakfield_f64_5", inputs, options.iterations, &reference, [](double x) {
            return sim_digamma_f64_5(x);
        }));
    rows.push_back(run_unary(
        "digamma",
        "oakfield_adaptive",
        inputs,
        options.iterations,
        &reference,
        [tol = options.adaptive_tolerance](double x) { return sim_digamma_f64_tail(x, tol); }));
    rows.push_back(run_unary(
        "digamma", "oakfield_mortici", inputs, options.iterations, &reference, [](double x) {
            return sim_digamma_f64_mortici(x);
        }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(
        run_unary("digamma", "boost_math", inputs, options.iterations, &reference, [](double x) {
            return boost::math::digamma(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(
        run_unary("digamma", "gsl", inputs, options.iterations, &reference, [](double x) {
            return gsl_sf_psi(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "digamma", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.digamma(x);
        })));
#endif

    print_group_results(options, "digamma", rows, reference_name());
}

static void run_trigamma_group(const std::vector<double>& inputs, const Options& options) {
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.trigamma(x); });
#else
    const std::vector<double> reference =
        evaluate_unary(inputs, [](double x) { return sim_trigamma_f64_12(x); });
#endif

    rows.push_back(run_unary(
        "trigamma", "oakfield_f64_12", inputs, options.iterations, &reference, [](double x) {
            return sim_trigamma_f64_12(x);
        }));
    rows.push_back(run_unary(
        "trigamma", "oakfield_f64_7", inputs, options.iterations, &reference, [](double x) {
            return sim_trigamma_f64_7(x);
        }));
    rows.push_back(run_unary(
        "trigamma", "oakfield_f64_5", inputs, options.iterations, &reference, [](double x) {
            return sim_trigamma_f64_5(x);
        }));
    rows.push_back(run_unary(
        "trigamma",
        "oakfield_adaptive",
        inputs,
        options.iterations,
        &reference,
        [tol = options.adaptive_tolerance](double x) { return sim_trigamma_f64_tail(x, tol); }));
    rows.push_back(run_unary(
        "trigamma", "oakfield_mortici", inputs, options.iterations, &reference, [](double x) {
            return sim_trigamma_f64_mortici(x);
        }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(
        run_unary("trigamma", "boost_math", inputs, options.iterations, &reference, [](double x) {
            return boost::math::polygamma(1, x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(
        run_unary("trigamma", "gsl", inputs, options.iterations, &reference, [](double x) {
            return gsl_sf_psi_1(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "trigamma", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.trigamma(x);
        })));
#endif

    print_group_results(options, "trigamma", rows, reference_name());
}

static void run_tetragamma_group(const std::vector<double>& inputs, const Options& options) {
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.tetragamma(x); });
#else
    const std::vector<double> reference =
        evaluate_unary(inputs, [](double x) { return sim_tetragamma_f64_12(x); });
#endif

    rows.push_back(run_unary(
        "tetragamma", "oakfield_f64_12", inputs, options.iterations, &reference, [](double x) {
            return sim_tetragamma_f64_12(x);
        }));
    rows.push_back(run_unary(
        "tetragamma", "oakfield_f64_7", inputs, options.iterations, &reference, [](double x) {
            return sim_tetragamma_f64_7(x);
        }));
    rows.push_back(run_unary(
        "tetragamma", "oakfield_f64_5", inputs, options.iterations, &reference, [](double x) {
            return sim_tetragamma_f64_5(x);
        }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(
        run_unary("tetragamma", "boost_math", inputs, options.iterations, &reference, [](double x) {
            return boost::math::polygamma(2, x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(
        run_unary("tetragamma", "gsl", inputs, options.iterations, &reference, [](double x) {
            return gsl_sf_psi_n(2, x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "tetragamma", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.tetragamma(x);
        })));
#endif

    print_group_results(options, "tetragamma", rows, reference_name());
}

static void run_complex_digamma_group(const std::vector<SimComplexDouble>& inputs,
                                      const Options&                       options) {
    std::vector<BenchmarkRow>           rows;
    const std::vector<SimComplexDouble> reference =
        evaluate_complex_unary(inputs, [](SimComplexDouble z) { return sim_digamma_c64_12(z); });

    rows.push_back(run_complex_unary("digamma_complex",
                                     "oakfield_c64_12",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_digamma_c64_12(z); }));
    rows.push_back(run_complex_unary("digamma_complex",
                                     "oakfield_c64_7",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_digamma_c64_7(z); }));
    rows.push_back(run_complex_unary("digamma_complex",
                                     "oakfield_c64_5",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_digamma_c64_5(z); }));
    rows.push_back(run_complex_unary("digamma_complex",
                                     "oakfield_adaptive",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [tol = options.adaptive_tolerance](SimComplexDouble z) {
                                         return sim_digamma_c64_tail(z, tol);
                                     }));
    rows.push_back(
        run_complex_unary("digamma_complex",
                          "oakfield_mortici",
                          inputs,
                          options.iterations,
                          &reference,
                          [](SimComplexDouble z) { return sim_digamma_c64_mortici(z); }));

    print_group_results(options, "digamma_complex", rows, "oakfield_c64_12");
}

static void run_complex_trigamma_group(const std::vector<SimComplexDouble>& inputs,
                                       const Options&                       options) {
    std::vector<BenchmarkRow>           rows;
    const std::vector<SimComplexDouble> reference =
        evaluate_complex_unary(inputs, [](SimComplexDouble z) { return sim_trigamma_c64_12(z); });

    rows.push_back(run_complex_unary("trigamma_complex",
                                     "oakfield_c64_12",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_trigamma_c64_12(z); }));
    rows.push_back(run_complex_unary("trigamma_complex",
                                     "oakfield_c64_7",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_trigamma_c64_7(z); }));
    rows.push_back(run_complex_unary("trigamma_complex",
                                     "oakfield_c64_5",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_trigamma_c64_5(z); }));
    rows.push_back(run_complex_unary("trigamma_complex",
                                     "oakfield_adaptive",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [tol = options.adaptive_tolerance](SimComplexDouble z) {
                                         return sim_trigamma_c64_tail(z, tol);
                                     }));

    print_group_results(options, "trigamma_complex", rows, "oakfield_c64_12");
}

static void run_complex_tetragamma_group(const std::vector<SimComplexDouble>& inputs,
                                         const Options&                       options) {
    std::vector<BenchmarkRow>           rows;
    const std::vector<SimComplexDouble> reference =
        evaluate_complex_unary(inputs, [](SimComplexDouble z) { return sim_tetragamma_c64_12(z); });

    rows.push_back(run_complex_unary("tetragamma_complex",
                                     "oakfield_c64_12",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_tetragamma_c64_12(z); }));
    rows.push_back(run_complex_unary("tetragamma_complex",
                                     "oakfield_c64_7",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_tetragamma_c64_7(z); }));
    rows.push_back(run_complex_unary("tetragamma_complex",
                                     "oakfield_c64_5",
                                     inputs,
                                     options.iterations,
                                     &reference,
                                     [](SimComplexDouble z) { return sim_tetragamma_c64_5(z); }));

    print_group_results(options, "tetragamma_complex", rows, "oakfield_c64_12");
}

static void run_airy_group(const std::vector<double>& inputs, const Options& options) {
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.airy_ai(x); });
#else
    const std::vector<double> reference =
        evaluate_unary(inputs, [](double x) { return sim_airy_ai_f64(x); });
#endif

    rows.push_back(
        run_unary("airy_ai", "oakfield", inputs, options.iterations, &reference, [](double x) {
            return sim_airy_ai_f64(x);
        }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(
        run_unary("airy_ai", "boost_math", inputs, options.iterations, &reference, [](double x) {
            return boost::math::airy_ai(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(
        run_unary("airy_ai", "gsl", inputs, options.iterations, &reference, [](double x) {
            return gsl_sf_airy_Ai(x, GSL_PREC_DOUBLE);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "airy_ai", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.airy_ai(x);
        })));
#endif

    print_group_results(options, "airy_ai", rows, reference_name());
}

static void run_bessel_group(const std::vector<double>& inputs, const Options& options) {
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_indexed(inputs, [&oracle](std::size_t index, double x) {
            return oracle.bessel_jn(bessel_order_for_index(index), x);
        });
#else
    const std::vector<double> reference = evaluate_indexed(inputs, [](std::size_t index, double x) {
        return sim_bessel_jn_f64(bessel_order_for_index(index), x);
    });
#endif

    rows.push_back(run_indexed("bessel_jn",
                               "oakfield",
                               inputs,
                               options.iterations,
                               &reference,
                               [](std::size_t index, double x) {
                                   return sim_bessel_jn_f64(bessel_order_for_index(index), x);
                               }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(run_indexed("bessel_jn",
                               "boost_math",
                               inputs,
                               options.iterations,
                               &reference,
                               [](std::size_t index, double x) {
                                   return boost::math::cyl_bessel_j(bessel_order_for_index(index),
                                                                    x);
                               }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(run_indexed("bessel_jn",
                               "gsl",
                               inputs,
                               options.iterations,
                               &reference,
                               [](std::size_t index, double x) {
                                   return gsl_sf_bessel_Jn(bessel_order_for_index(index), x);
                               }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_indexed("bessel_jn",
                                           "mpfr",
                                           inputs,
                                           options.mpfr_iterations,
                                           &reference,
                                           [&oracle](std::size_t index, double x) {
                                               return oracle.bessel_jn(
                                                   bessel_order_for_index(index), x);
                                           })));
#endif

    print_group_results(options, "bessel_jn", rows, reference_name());
}

static double oakfield_real_zeta(double x) {
#if OAKFIELD_ENABLE_ZETA_CORE
    SimComplexDouble s       = { x, 0.0 };
    SimZetaContext   context = sim_zeta_context_interactive();
    SimZetaResult    result  = sim_zeta_eval(s, &context);
    if (result.status != SIM_ZETA_STATUS_OK) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return result.value.re;
#else
    (void) x;
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

static SimComplexDouble oakfield_complex_zeta(SimComplexDouble s, const SimZetaContext* context) {
#if OAKFIELD_ENABLE_ZETA_CORE
    SimZetaResult result = sim_zeta_eval(s, context);
    if (result.status != SIM_ZETA_STATUS_OK) {
        return (SimComplexDouble) { std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN() };
    }
    return result.value;
#else
    (void) s;
    (void) context;
    return (SimComplexDouble) { std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN() };
#endif
}

static SimComplexDouble oakfield_complex_xi(SimComplexDouble s, const SimXiContext* context) {
#if OAKFIELD_ENABLE_ZETA_CORE
    SimXiResult result = sim_xi_eval(s, context);
    if (result.status != SIM_ZETA_STATUS_OK) {
        return (SimComplexDouble) { std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN() };
    }
    return result.value;
#else
    (void) s;
    (void) context;
    return (SimComplexDouble) { std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN() };
#endif
}

static void run_zeta_group(const std::vector<double>& inputs, const Options& options) {
#if OAKFIELD_ENABLE_ZETA_CORE
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.zeta(x); });
#else
    const std::vector<double> reference =
        evaluate_unary(inputs, [](double x) { return oakfield_real_zeta(x); });
#endif

    rows.push_back(run_unary("zeta_real",
                             "oakfield_interactive",
                             inputs,
                             options.zeta_iterations,
                             &reference,
                             [](double x) { return oakfield_real_zeta(x); }));

#if OAKFIELD_BENCH_HAVE_BOOST_MATH
    rows.push_back(run_unary(
        "zeta_real", "boost_math", inputs, options.zeta_iterations, &reference, [](double x) {
            return boost::math::zeta(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_GSL
    rows.push_back(
        run_unary("zeta_real", "gsl", inputs, options.zeta_iterations, &reference, [](double x) {
            return gsl_sf_zeta(x);
        }));
#endif

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "zeta_real", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.zeta(x);
        })));
#endif

    print_group_results(options, "zeta_real", rows, reference_name());
#else
    std::vector<BenchmarkRow> rows;
    (void) inputs;
    rows.push_back(make_skipped_row("zeta_real", "oakfield_interactive", "zeta_core_off"));
    print_group_results(options, "zeta_real", rows, "none");
#endif
}

static void run_xi_group(const std::vector<double>& inputs, const Options& options) {
#if OAKFIELD_ENABLE_ZETA_CORE
    std::vector<BenchmarkRow> rows;
#if OAKFIELD_BENCH_HAVE_MPFR
    MpfrWorkspace             oracle(options.mpfr_precision_bits);
    const std::vector<double> reference =
        evaluate_unary(inputs, [&oracle](double x) { return oracle.xi(x); });
#else
    SimXiContext              default_context = sim_xi_context_default();
    const std::vector<double> reference = evaluate_unary(inputs, [&default_context](double x) {
        return oakfield_complex_xi((SimComplexDouble) { x, 0.0 }, &default_context).re;
    });
#endif

    SimXiContext interactive_context = sim_xi_context_interactive();
#if OAKFIELD_BENCH_HAVE_MPFR
    SimXiContext default_context = sim_xi_context_default();
#endif
    rows.push_back(run_unary(
        "xi_real",
        "oakfield_interactive",
        inputs,
        options.zeta_iterations,
        &reference,
        [&interactive_context](double x) {
            return oakfield_complex_xi((SimComplexDouble) { x, 0.0 }, &interactive_context).re;
        }));
    rows.push_back(run_unary(
        "xi_real",
        "oakfield_default",
        inputs,
        options.zeta_iterations,
        &reference,
        [&default_context](double x) {
            return oakfield_complex_xi((SimComplexDouble) { x, 0.0 }, &default_context).re;
        }));

#if OAKFIELD_BENCH_HAVE_MPFR
    rows.push_back(mark_oracle(run_unary(
        "xi_real", "mpfr", inputs, options.mpfr_iterations, &reference, [&oracle](double x) {
            return oracle.xi(x);
        })));
#endif

    print_group_results(options, "xi_real", rows, reference_name());
#else
    std::vector<BenchmarkRow> rows;
    (void) inputs;
    rows.push_back(make_skipped_row("xi_real", "oakfield_interactive", "zeta_core_off"));
    print_group_results(options, "xi_real", rows, "none");
#endif
}

static void run_zeta_complex_group(const std::vector<SimComplexDouble>& inputs,
                                   const Options&                       options) {
#if OAKFIELD_ENABLE_ZETA_CORE
    SimZetaContext                      default_context     = sim_zeta_context_default();
    SimZetaContext                      interactive_context = sim_zeta_context_interactive();
    std::vector<BenchmarkRow>           rows;
    const std::vector<SimComplexDouble> reference =
        evaluate_complex_unary(inputs, [&default_context](SimComplexDouble s) {
            return oakfield_complex_zeta(s, &default_context);
        });

    rows.push_back(run_complex_unary("zeta_complex",
                                     "oakfield_interactive",
                                     inputs,
                                     options.zeta_iterations,
                                     &reference,
                                     [&interactive_context](SimComplexDouble s) {
                                         return oakfield_complex_zeta(s, &interactive_context);
                                     }));
    rows.push_back(mark_oracle(run_complex_unary("zeta_complex",
                                                 "oakfield_default",
                                                 inputs,
                                                 options.zeta_iterations,
                                                 &reference,
                                                 [&default_context](SimComplexDouble s) {
                                                     return oakfield_complex_zeta(s,
                                                                                  &default_context);
                                                 })));

    print_group_results(options, "zeta_complex", rows, "oakfield_default");
#else
    std::vector<BenchmarkRow> rows;
    (void) inputs;
    rows.push_back(make_skipped_row("zeta_complex", "oakfield_interactive", "zeta_core_off"));
    print_group_results(options, "zeta_complex", rows, "none");
#endif
}

static void run_xi_complex_group(const std::vector<SimComplexDouble>& inputs,
                                 const Options&                       options) {
#if OAKFIELD_ENABLE_ZETA_CORE
    SimXiContext                        default_context     = sim_xi_context_default();
    SimXiContext                        interactive_context = sim_xi_context_interactive();
    std::vector<BenchmarkRow>           rows;
    const std::vector<SimComplexDouble> reference =
        evaluate_complex_unary(inputs, [&default_context](SimComplexDouble s) {
            return oakfield_complex_xi(s, &default_context);
        });

    rows.push_back(run_complex_unary("xi_complex",
                                     "oakfield_interactive",
                                     inputs,
                                     options.zeta_iterations,
                                     &reference,
                                     [&interactive_context](SimComplexDouble s) {
                                         return oakfield_complex_xi(s, &interactive_context);
                                     }));
    rows.push_back(mark_oracle(run_complex_unary("xi_complex",
                                                 "oakfield_default",
                                                 inputs,
                                                 options.zeta_iterations,
                                                 &reference,
                                                 [&default_context](SimComplexDouble s) {
                                                     return oakfield_complex_xi(s,
                                                                                &default_context);
                                                 })));

    print_group_results(options, "xi_complex", rows, "oakfield_default");
#else
    std::vector<BenchmarkRow> rows;
    (void) inputs;
    rows.push_back(make_skipped_row("xi_complex", "oakfield_interactive", "zeta_core_off"));
    print_group_results(options, "xi_complex", rows, "none");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Options   options;
    const int parse_result = parse_options(argc, argv, &options);

    if (parse_result > 0) {
        return 0;
    }
    if (parse_result < 0) {
        return 1;
    }

#if OAKFIELD_BENCH_HAVE_GSL
    gsl_set_error_handler_off();
#endif

    const std::vector<double>           positive_inputs = make_positive_inputs(options.samples);
    const std::vector<double>           airy_inputs     = make_airy_inputs(options.samples);
    const std::vector<double>           bessel_inputs   = make_bessel_inputs(options.samples);
    const std::vector<double>           zeta_inputs     = make_zeta_inputs(options.samples);
    const std::vector<SimComplexDouble> special_complex_inputs =
        make_special_complex_inputs(options.samples);
    const std::vector<SimComplexDouble> zeta_complex_inputs =
        make_zeta_complex_inputs(options.samples);

    if (options.key_value_output) {
        std::printf("benchmark_special_functions iterations=%d samples=%d "
                    "zeta_iterations=%d mpfr_iterations=%d mpfr_precision_bits=%d "
                    "adaptive_tolerance=%.3e format=kv\n",
                    options.iterations,
                    options.samples,
                    options.zeta_iterations,
                    options.mpfr_iterations,
                    options.mpfr_precision_bits,
                    options.adaptive_tolerance);
        std::printf("suggested_comparison_libraries=GSL,Boost.Math,MPFR\n");
        std::printf("library_availability boost_math=%d gsl=%d mpfr=%d zeta_core=%d\n",
                    OAKFIELD_BENCH_HAVE_BOOST_MATH,
                    OAKFIELD_BENCH_HAVE_GSL,
                    OAKFIELD_BENCH_HAVE_MPFR,
                    OAKFIELD_ENABLE_ZETA_CORE);
        std::printf("primary_real_reference=%s\n", reference_name());
        std::printf("Delta columns compare each row to the row-specific reference field.\n");
    } else {
        std::printf("benchmark_special_functions\n");
        std::printf("  iterations: %d\n", options.iterations);
        std::printf("  samples: %d\n", options.samples);
        std::printf("  zeta iterations: %d\n", options.zeta_iterations);
        std::printf("  MPFR iterations: %d\n", options.mpfr_iterations);
        std::printf("  MPFR precision: %d bits\n", options.mpfr_precision_bits);
        std::printf("  adaptive tolerance: %.3e\n", options.adaptive_tolerance);
        std::printf("  optional libraries: Boost.Math=%s, GSL=%s, MPFR=%s, zeta_core=%s\n",
                    OAKFIELD_BENCH_HAVE_BOOST_MATH ? "yes" : "no",
                    OAKFIELD_BENCH_HAVE_GSL ? "yes" : "no",
                    OAKFIELD_BENCH_HAVE_MPFR ? "yes" : "no",
                    OAKFIELD_ENABLE_ZETA_CORE ? "yes" : "no");
        std::printf("  primary real reference: %s\n", reference_name());
        std::printf("  deltas: compared to the per-group reference named in each section\n");
        std::printf("  markers: fastest is green, most accurate non-oracle is cyan when color is "
                    "enabled\n");
    }

    run_digamma_group(positive_inputs, options);
    run_trigamma_group(positive_inputs, options);
    run_tetragamma_group(positive_inputs, options);
    run_complex_digamma_group(special_complex_inputs, options);
    run_complex_trigamma_group(special_complex_inputs, options);
    run_complex_tetragamma_group(special_complex_inputs, options);
    run_airy_group(airy_inputs, options);
    run_bessel_group(bessel_inputs, options);
    if (options.include_zeta) {
        run_zeta_group(zeta_inputs, options);
        run_xi_group(zeta_inputs, options);
        run_zeta_complex_group(zeta_complex_inputs, options);
        run_xi_complex_group(zeta_complex_inputs, options);
    }

    if (g_sink == -1.0) {
        std::printf("sink=%.1f\n", g_sink);
    }

    return 0;
}
