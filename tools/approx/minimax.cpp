#include <mpfr.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class FunctionId {
    digamma,
    airy_ai,
    bessel_jn,
};

enum class WeightMode {
    absolute,
    relative,
};

enum class OutputFormat {
    summary,
    c,
    json,
};

struct Options {
    FunctionId function = FunctionId::digamma;
    WeightMode weight = WeightMode::absolute;
    OutputFormat format = OutputFormat::summary;
    double a = 1.0;
    double b = 2.0;
    int num_degree = 5;
    int den_degree = 0;
    int order = 0;
    int grid = 8192;
    int iterations = 8;
    int precision_bits = 256;
    double relative_floor = 1.0e-300;
};

struct Candidate {
    double x = 0.0;
    double error = 0.0;
};

struct SolveResult {
    std::vector<long double> p;
    std::vector<long double> q;
    long double e = 0.0L;
};

struct ErrorStats {
    long double max_abs = 0.0L;
    long double rms = 0.0L;
    double x_at_max = 0.0;
};

bool starts_with(const char* value, const char* prefix) {
    return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

const char* function_name(FunctionId id) {
    switch (id) {
    case FunctionId::digamma:
        return "digamma";
    case FunctionId::airy_ai:
        return "airy_ai";
    case FunctionId::bessel_jn:
        return "bessel_jn";
    }
    return "unknown";
}

const char* weight_name(WeightMode mode) {
    return mode == WeightMode::relative ? "relative" : "absolute";
}

double parse_double_value(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
        std::ostringstream message;
        message << "invalid " << name << ": " << text;
        throw std::runtime_error(message.str());
    }
    return value;
}

int parse_int_value(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        std::ostringstream message;
        message << "invalid " << name << ": " << text;
        throw std::runtime_error(message.str());
    }
    return static_cast<int>(value);
}

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " [options]\n"
        << "  --function=digamma|airy_ai|bessel_jn\n"
        << "  --a=X --b=Y\n"
        << "  --num-degree=N --den-degree=N\n"
        << "  --order=N                    bessel_jn order\n"
        << "  --weight=absolute|relative\n"
        << "  --relative-floor=X\n"
        << "  --grid=N --iterations=N --precision=BITS\n"
        << "  --format=summary|c|json\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (starts_with(arg, "--function=")) {
            const char* value = arg + std::strlen("--function=");
            if (std::strcmp(value, "digamma") == 0) {
                options.function = FunctionId::digamma;
            } else if (std::strcmp(value, "airy_ai") == 0) {
                options.function = FunctionId::airy_ai;
            } else if (std::strcmp(value, "bessel_jn") == 0) {
                options.function = FunctionId::bessel_jn;
            } else {
                throw std::runtime_error("unknown function");
            }
        } else if (starts_with(arg, "--a=")) {
            options.a = parse_double_value(arg + std::strlen("--a="), "--a");
        } else if (starts_with(arg, "--b=")) {
            options.b = parse_double_value(arg + std::strlen("--b="), "--b");
        } else if (starts_with(arg, "--num-degree=")) {
            options.num_degree = parse_int_value(arg + std::strlen("--num-degree="),
                                                 "--num-degree");
        } else if (starts_with(arg, "--den-degree=")) {
            options.den_degree = parse_int_value(arg + std::strlen("--den-degree="),
                                                 "--den-degree");
        } else if (starts_with(arg, "--order=")) {
            options.order = parse_int_value(arg + std::strlen("--order="), "--order");
        } else if (starts_with(arg, "--grid=")) {
            options.grid = parse_int_value(arg + std::strlen("--grid="), "--grid");
        } else if (starts_with(arg, "--iterations=")) {
            options.iterations = parse_int_value(arg + std::strlen("--iterations="),
                                                 "--iterations");
        } else if (starts_with(arg, "--precision=")) {
            options.precision_bits = parse_int_value(arg + std::strlen("--precision="),
                                                     "--precision");
        } else if (starts_with(arg, "--relative-floor=")) {
            options.relative_floor = parse_double_value(arg + std::strlen("--relative-floor="),
                                                        "--relative-floor");
        } else if (starts_with(arg, "--weight=")) {
            const char* value = arg + std::strlen("--weight=");
            if (std::strcmp(value, "absolute") == 0) {
                options.weight = WeightMode::absolute;
            } else if (std::strcmp(value, "relative") == 0) {
                options.weight = WeightMode::relative;
            } else {
                throw std::runtime_error("unknown weight");
            }
        } else if (starts_with(arg, "--format=")) {
            const char* value = arg + std::strlen("--format=");
            if (std::strcmp(value, "summary") == 0) {
                options.format = OutputFormat::summary;
            } else if (std::strcmp(value, "c") == 0) {
                options.format = OutputFormat::c;
            } else if (std::strcmp(value, "json") == 0) {
                options.format = OutputFormat::json;
            } else {
                throw std::runtime_error("unknown output format");
            }
        } else {
            std::ostringstream message;
            message << "unknown option: " << arg;
            throw std::runtime_error(message.str());
        }
    }

    if (!(options.a < options.b)) {
        throw std::runtime_error("--a must be less than --b");
    }
    if (options.num_degree < 0 || options.den_degree < 0) {
        throw std::runtime_error("degrees must be nonnegative");
    }
    if (options.num_degree + options.den_degree + 2 < 2) {
        throw std::runtime_error("invalid unknown count");
    }
    if (options.grid < options.num_degree + options.den_degree + 3) {
        throw std::runtime_error("--grid is too small for the requested degrees");
    }
    if (options.iterations < 1) {
        throw std::runtime_error("--iterations must be positive");
    }
    if (options.precision_bits < 80) {
        throw std::runtime_error("--precision must be at least 80 bits");
    }
    if (!(options.relative_floor > 0.0)) {
        throw std::runtime_error("--relative-floor must be positive");
    }
    return options;
}

class MpfrOracle {
public:
    explicit MpfrOracle(const Options& options)
        : options_(options) {
        mpfr_init2(input_, static_cast<mpfr_prec_t>(options.precision_bits));
        mpfr_init2(output_, static_cast<mpfr_prec_t>(options.precision_bits));
    }

    MpfrOracle(const MpfrOracle&) = delete;
    MpfrOracle& operator=(const MpfrOracle&) = delete;

    ~MpfrOracle() {
        mpfr_clear(output_);
        mpfr_clear(input_);
    }

    long double eval(double x) {
        mpfr_set_d(input_, x, MPFR_RNDN);
        switch (options_.function) {
        case FunctionId::digamma:
            mpfr_digamma(output_, input_, MPFR_RNDN);
            break;
        case FunctionId::airy_ai:
            mpfr_ai(output_, input_, MPFR_RNDN);
            break;
        case FunctionId::bessel_jn:
            mpfr_jn(output_, static_cast<long>(options_.order), input_, MPFR_RNDN);
            break;
        }
        return mpfr_get_ld(output_, MPFR_RNDN);
    }

private:
    const Options& options_;
    mpfr_t input_;
    mpfr_t output_;
};

long double map_to_chebyshev(double x, const Options& options) {
    return (2.0L * static_cast<long double>(x) - static_cast<long double>(options.a) -
            static_cast<long double>(options.b)) /
           (static_cast<long double>(options.b) - static_cast<long double>(options.a));
}

std::vector<long double> chebyshev_values(long double t, int degree) {
    std::vector<long double> values(static_cast<std::size_t>(degree + 1), 0.0L);
    if (degree < 0) {
        return values;
    }
    values[0] = 1.0L;
    if (degree >= 1) {
        values[1] = t;
    }
    for (int i = 2; i <= degree; ++i) {
        values[static_cast<std::size_t>(i)] =
            2.0L * t * values[static_cast<std::size_t>(i - 1)] -
            values[static_cast<std::size_t>(i - 2)];
    }
    return values;
}

long double weight_scale(long double f, const Options& options) {
    if (options.weight == WeightMode::absolute) {
        return 1.0L;
    }
    return std::max(std::fabsl(f), static_cast<long double>(options.relative_floor));
}

std::vector<double> initial_points(const Options& options) {
    constexpr long double pi = 3.141592653589793238462643383279502884L;
    const int count = options.num_degree + options.den_degree + 2;
    std::vector<double> points;
    points.reserve(static_cast<std::size_t>(count));
    if (count == 1) {
        points.push_back(0.5 * (options.a + options.b));
        return points;
    }
    for (int i = 0; i < count; ++i) {
        const long double theta = pi * static_cast<long double>(count - 1 - i) /
                                  static_cast<long double>(count - 1);
        const long double u = 0.5L * (1.0L + std::cos(theta));
        points.push_back(static_cast<double>(static_cast<long double>(options.a) +
                                             (static_cast<long double>(options.b) -
                                              static_cast<long double>(options.a)) *
                                                 u));
    }
    std::sort(points.begin(), points.end());
    return points;
}

std::vector<long double> solve_linear_system(std::vector<std::vector<long double>> matrix) {
    const std::size_t n = matrix.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        long double pivot_abs = std::fabsl(matrix[col][col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const long double value_abs = std::fabsl(matrix[row][col]);
            if (value_abs > pivot_abs) {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if (pivot_abs <= 1.0e-30L) {
            throw std::runtime_error("singular linear system while solving approximant");
        }
        if (pivot != col) {
            std::swap(matrix[pivot], matrix[col]);
        }
        const long double inv_pivot = 1.0L / matrix[col][col];
        for (std::size_t j = col; j <= n; ++j) {
            matrix[col][j] *= inv_pivot;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const long double factor = matrix[row][col];
            if (factor == 0.0L) {
                continue;
            }
            for (std::size_t j = col; j <= n; ++j) {
                matrix[row][j] -= factor * matrix[col][j];
            }
        }
    }

    std::vector<long double> solution(n);
    for (std::size_t row = 0; row < n; ++row) {
        solution[row] = matrix[row][n];
    }
    return solution;
}

SolveResult solve_approximant(const std::vector<double>& points,
                              const std::vector<long double>& values,
                              const Options& options) {
    const int p_count = options.num_degree + 1;
    const int q_count = options.den_degree;
    const int unknowns = p_count + q_count + 1;
    if (static_cast<int>(points.size()) != unknowns ||
        static_cast<int>(values.size()) != unknowns) {
        throw std::runtime_error("internal alternant size mismatch");
    }

    std::vector<std::vector<long double>> matrix(
        static_cast<std::size_t>(unknowns),
        std::vector<long double>(static_cast<std::size_t>(unknowns + 1), 0.0L));

    for (int row = 0; row < unknowns; ++row) {
        const long double t = map_to_chebyshev(points[static_cast<std::size_t>(row)], options);
        const std::vector<long double> p_basis = chebyshev_values(t, options.num_degree);
        const std::vector<long double> q_basis = chebyshev_values(t, options.den_degree);
        const long double f = values[static_cast<std::size_t>(row)];
        int col = 0;
        for (int j = 0; j < p_count; ++j) {
            matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col++)] =
                p_basis[static_cast<std::size_t>(j)];
        }
        for (int j = 1; j <= q_count; ++j) {
            matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col++)] =
                -f * q_basis[static_cast<std::size_t>(j)];
        }
        const long double sign = (row & 1) ? -1.0L : 1.0L;
        matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
            sign * weight_scale(f, options);
        matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(unknowns)] = f;
    }

    const std::vector<long double> solution = solve_linear_system(std::move(matrix));
    SolveResult result;
    result.p.assign(solution.begin(), solution.begin() + p_count);
    result.q.assign(solution.begin() + p_count, solution.begin() + p_count + q_count);
    result.e = solution.back();
    return result;
}

long double eval_chebyshev_series(const std::vector<long double>& coefficients, long double t) {
    if (coefficients.empty()) {
        return 0.0L;
    }
    long double sum = coefficients[0];
    if (coefficients.size() == 1U) {
        return sum;
    }
    long double t_prev = 1.0L;
    long double t_cur = t;
    sum += coefficients[1] * t_cur;
    for (std::size_t i = 2; i < coefficients.size(); ++i) {
        const long double t_next = 2.0L * t * t_cur - t_prev;
        sum += coefficients[i] * t_next;
        t_prev = t_cur;
        t_cur = t_next;
    }
    return sum;
}

long double eval_approximation(const SolveResult& approx, double x, const Options& options) {
    const long double t = map_to_chebyshev(x, options);
    const long double p = eval_chebyshev_series(approx.p, t);
    std::vector<long double> q_full;
    q_full.reserve(approx.q.size() + 1U);
    q_full.push_back(1.0L);
    q_full.insert(q_full.end(), approx.q.begin(), approx.q.end());
    const long double q = eval_chebyshev_series(q_full, t);
    return p / q;
}

std::vector<long double> oracle_values(const std::vector<double>& points, MpfrOracle& oracle) {
    std::vector<long double> values;
    values.reserve(points.size());
    for (double x : points) {
        values.push_back(oracle.eval(x));
    }
    return values;
}

std::vector<Candidate> dense_errors(const SolveResult& approx, const Options& options,
                                    MpfrOracle& oracle) {
    std::vector<Candidate> errors;
    errors.reserve(static_cast<std::size_t>(options.grid));
    for (int i = 0; i < options.grid; ++i) {
        const double u = (options.grid == 1) ? 0.5 : static_cast<double>(i) /
                                                          static_cast<double>(options.grid - 1);
        const double x = options.a + (options.b - options.a) * u;
        const long double f = oracle.eval(x);
        const long double r = eval_approximation(approx, x, options);
        const long double scale = weight_scale(f, options);
        errors.push_back({ x, static_cast<double>((r - f) / scale) });
    }
    return errors;
}

ErrorStats error_stats(const std::vector<Candidate>& errors) {
    ErrorStats stats;
    long double sum_sq = 0.0L;
    for (const Candidate& e : errors) {
        const long double abs_error = std::fabsl(static_cast<long double>(e.error));
        sum_sq += static_cast<long double>(e.error) * static_cast<long double>(e.error);
        if (abs_error > stats.max_abs) {
            stats.max_abs = abs_error;
            stats.x_at_max = e.x;
        }
    }
    if (!errors.empty()) {
        stats.rms = std::sqrt(sum_sq / static_cast<long double>(errors.size()));
    }
    return stats;
}

int sign_of(double value) {
    if (value > 0.0) {
        return 1;
    }
    if (value < 0.0) {
        return -1;
    }
    return 0;
}

std::vector<double> select_alternants(const std::vector<Candidate>& errors, int count,
                                      const Options& options) {
    std::vector<Candidate> candidates;
    if (errors.empty()) {
        return initial_points(options);
    }

    candidates.push_back(errors.front());
    for (std::size_t i = 1; i + 1 < errors.size(); ++i) {
        const double a = std::fabs(errors[i - 1].error);
        const double b = std::fabs(errors[i].error);
        const double c = std::fabs(errors[i + 1].error);
        if (b >= a && b >= c) {
            candidates.push_back(errors[i]);
        }
    }
    candidates.push_back(errors.back());
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.x < rhs.x; });

    std::vector<Candidate> alternating;
    for (const Candidate& candidate : candidates) {
        const int s = sign_of(candidate.error);
        if (s == 0) {
            continue;
        }
        if (alternating.empty()) {
            alternating.push_back(candidate);
            continue;
        }
        const int last_sign = sign_of(alternating.back().error);
        if (last_sign != s) {
            alternating.push_back(candidate);
        } else if (std::fabs(candidate.error) > std::fabs(alternating.back().error)) {
            alternating.back() = candidate;
        }
    }

    while (static_cast<int>(alternating.size()) > count) {
        auto remove_it = alternating.begin();
        long double smallest = std::numeric_limits<long double>::infinity();
        for (auto it = alternating.begin(); it != alternating.end(); ++it) {
            const long double value = std::fabsl(static_cast<long double>(it->error));
            if (value < smallest) {
                smallest = value;
                remove_it = it;
            }
        }
        alternating.erase(remove_it);
    }

    if (static_cast<int>(alternating.size()) < count) {
        std::vector<Candidate> by_error = candidates;
        std::sort(by_error.begin(), by_error.end(), [](const Candidate& lhs,
                                                       const Candidate& rhs) {
            return std::fabs(lhs.error) > std::fabs(rhs.error);
        });
        for (const Candidate& candidate : by_error) {
            if (static_cast<int>(alternating.size()) >= count) {
                break;
            }
            const auto close = std::find_if(alternating.begin(), alternating.end(),
                                            [&](const Candidate& existing) {
                                                return std::fabs(existing.x - candidate.x) <=
                                                       1.0e-14 *
                                                           std::max(1.0, std::fabs(candidate.x));
                                            });
            if (close == alternating.end()) {
                alternating.push_back(candidate);
            }
        }
    }

    std::sort(alternating.begin(), alternating.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.x < rhs.x; });
    if (static_cast<int>(alternating.size()) > count) {
        alternating.resize(static_cast<std::size_t>(count));
    }

    std::vector<double> points;
    points.reserve(static_cast<std::size_t>(count));
    for (const Candidate& candidate : alternating) {
        points.push_back(candidate.x);
    }
    if (static_cast<int>(points.size()) != count) {
        return initial_points(options);
    }
    return points;
}

void emit_vector_c(const char* name, const std::vector<long double>& values) {
    std::cout << "static const double " << name << "[" << values.size() << "] = {\n";
    std::cout << std::setprecision(17);
    for (long double value : values) {
        std::cout << "    " << static_cast<double>(value) << ",\n";
    }
    std::cout << "};\n";
}

void emit_summary(const Options& options, const SolveResult& approx, const ErrorStats& stats) {
    std::cout << "function=" << function_name(options.function) << "\n";
    if (options.function == FunctionId::bessel_jn) {
        std::cout << "order=" << options.order << "\n";
    }
    std::cout << std::setprecision(17);
    std::cout << "interval=[" << options.a << ", " << options.b << "]\n";
    std::cout << "basis=chebyshev\n";
    std::cout << "num_degree=" << options.num_degree << "\n";
    std::cout << "den_degree=" << options.den_degree << "\n";
    std::cout << "weight=" << weight_name(options.weight) << "\n";
    std::cout << "estimated_error=" << static_cast<double>(approx.e) << "\n";
    std::cout << "grid_max_abs_error=" << static_cast<double>(stats.max_abs) << "\n";
    std::cout << "grid_rms_error=" << static_cast<double>(stats.rms) << "\n";
    std::cout << "grid_x_at_max=" << stats.x_at_max << "\n";
}

void emit_c(const Options& options, const SolveResult& approx, const ErrorStats& stats) {
    std::cout << "/* Generated by tools/approx/oakfield_minimax.\n";
    std::cout << " * function: " << function_name(options.function);
    if (options.function == FunctionId::bessel_jn) {
        std::cout << " order=" << options.order;
    }
    std::cout << "\n * interval: [" << std::setprecision(17) << options.a << ", " << options.b
              << "]\n";
    std::cout << " * numerator degree: " << options.num_degree << "\n";
    std::cout << " * denominator degree: " << options.den_degree << "\n";
    std::cout << " * weight: " << weight_name(options.weight) << "\n";
    std::cout << " * dense-grid max weighted error: " << static_cast<double>(stats.max_abs)
              << " at x=" << stats.x_at_max << "\n";
    std::cout << " */\n";
    std::vector<long double> q_full;
    q_full.reserve(approx.q.size() + 1U);
    q_full.push_back(1.0L);
    q_full.insert(q_full.end(), approx.q.begin(), approx.q.end());
    emit_vector_c("oakfield_minimax_p", approx.p);
    emit_vector_c("oakfield_minimax_q", q_full);
    std::cout << "static inline double oakfield_minimax_eval_cheb(const double* c, int n, "
                 "double t) {\n";
    std::cout << "    double sum = c[0];\n";
    std::cout << "    if (n == 1) return sum;\n";
    std::cout << "    double t_prev = 1.0;\n";
    std::cout << "    double t_cur = t;\n";
    std::cout << "    sum += c[1] * t_cur;\n";
    std::cout << "    for (int i = 2; i < n; ++i) {\n";
    std::cout << "        double t_next = 2.0 * t * t_cur - t_prev;\n";
    std::cout << "        sum += c[i] * t_next;\n";
    std::cout << "        t_prev = t_cur;\n";
    std::cout << "        t_cur = t_next;\n";
    std::cout << "    }\n";
    std::cout << "    return sum;\n";
    std::cout << "}\n";
    std::cout << "static inline double oakfield_minimax_eval(double x) {\n";
    std::cout << "    const double t = (2.0 * x - (" << options.a << ") - (" << options.b
              << ")) / ((" << options.b << ") - (" << options.a << "));\n";
    std::cout << "    const double p = oakfield_minimax_eval_cheb(oakfield_minimax_p, "
              << approx.p.size() << ", t);\n";
    std::cout << "    const double q = oakfield_minimax_eval_cheb(oakfield_minimax_q, "
              << q_full.size() << ", t);\n";
    std::cout << "    return p / q;\n";
    std::cout << "}\n";
}

void emit_json(const Options& options, const SolveResult& approx, const ErrorStats& stats) {
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"function\": \"" << function_name(options.function) << "\",\n";
    std::cout << "  \"order\": " << options.order << ",\n";
    std::cout << "  \"a\": " << options.a << ",\n";
    std::cout << "  \"b\": " << options.b << ",\n";
    std::cout << "  \"basis\": \"chebyshev\",\n";
    std::cout << "  \"num_degree\": " << options.num_degree << ",\n";
    std::cout << "  \"den_degree\": " << options.den_degree << ",\n";
    std::cout << "  \"weight\": \"" << weight_name(options.weight) << "\",\n";
    std::cout << "  \"grid_max_abs_error\": " << static_cast<double>(stats.max_abs) << ",\n";
    std::cout << "  \"grid_rms_error\": " << static_cast<double>(stats.rms) << ",\n";
    std::cout << "  \"p\": [";
    for (std::size_t i = 0; i < approx.p.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << static_cast<double>(approx.p[i]);
    }
    std::cout << "],\n";
    std::cout << "  \"q_tail\": [";
    for (std::size_t i = 0; i < approx.q.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << static_cast<double>(approx.q[i]);
    }
    std::cout << "]\n";
    std::cout << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        MpfrOracle oracle(options);
        std::vector<double> points = initial_points(options);
        SolveResult approx;
        SolveResult best_approx;
        ErrorStats best_stats;
        best_stats.max_abs = std::numeric_limits<long double>::infinity();
        std::vector<Candidate> errors;

        for (int iter = 0; iter < options.iterations; ++iter) {
            const std::vector<long double> values = oracle_values(points, oracle);
            approx = solve_approximant(points, values, options);
            errors = dense_errors(approx, options, oracle);
            const ErrorStats stats = error_stats(errors);
            if (stats.max_abs < best_stats.max_abs) {
                best_stats = stats;
                best_approx = approx;
            }
            points = select_alternants(errors, options.num_degree + options.den_degree + 2,
                                       options);
        }

        const std::vector<long double> values = oracle_values(points, oracle);
        approx = solve_approximant(points, values, options);
        errors = dense_errors(approx, options, oracle);
        const ErrorStats stats = error_stats(errors);
        if (stats.max_abs < best_stats.max_abs) {
            best_stats = stats;
            best_approx = approx;
        }

        switch (options.format) {
        case OutputFormat::summary:
            emit_summary(options, best_approx, best_stats);
            break;
        case OutputFormat::c:
            emit_c(options, best_approx, best_stats);
            break;
        case OutputFormat::json:
            emit_json(options, best_approx, best_stats);
            break;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "oakfield_minimax: " << ex.what() << "\n";
        return 1;
    }
}
