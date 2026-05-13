function(oakfield_add_benchmark name source)
    add_executable(${name} ${source})
    oakfield_suppress_duplicate_library_warnings(${name})
    target_compile_definitions(${name}
        PRIVATE
            OAKFIELD_BENCH_OAKFIELD_ENABLE_FAST_MATH=$<BOOL:${OAKFIELD_ENABLE_FAST_MATH}>)
    target_link_libraries(${name} PRIVATE simcore)
endfunction()

include(CheckIncludeFileCXX)
find_package(GSL QUIET)
find_package(PkgConfig QUIET)

set(OAKFIELD_BENCH_BOOST_INCLUDE_DIR "" CACHE PATH
    "Optional include directory containing boost/, or the boost/ directory itself, for special-function benchmarks")

set(_oakfield_bench_boost_include_candidates)
if(OAKFIELD_BENCH_BOOST_INCLUDE_DIR)
    list(APPEND _oakfield_bench_boost_include_candidates
        "${OAKFIELD_BENCH_BOOST_INCLUDE_DIR}")
else()
    file(GLOB _oakfield_bench_homebrew_boost_include_dirs
        LIST_DIRECTORIES true
        "/opt/homebrew/Cellar/boost/*/include"
        "/usr/local/Cellar/boost/*/include")
    if(_oakfield_bench_homebrew_boost_include_dirs)
        list(SORT _oakfield_bench_homebrew_boost_include_dirs
            COMPARE NATURAL
            ORDER DESCENDING)
    endif()
    list(APPEND _oakfield_bench_boost_include_candidates
        /opt/homebrew/include
        /usr/local/include
        /usr/include
        ${_oakfield_bench_homebrew_boost_include_dirs})
endif()

set(_oakfield_bench_boost_include_dir "")
foreach(_oakfield_bench_boost_candidate IN LISTS _oakfield_bench_boost_include_candidates)
    if(EXISTS "${_oakfield_bench_boost_candidate}/boost/math/special_functions/digamma.hpp")
        set(_oakfield_bench_boost_include_dir "${_oakfield_bench_boost_candidate}")
        break()
    elseif(EXISTS "${_oakfield_bench_boost_candidate}/math/special_functions/digamma.hpp")
        get_filename_component(_oakfield_bench_boost_include_dir
            "${_oakfield_bench_boost_candidate}" DIRECTORY)
        break()
    endif()
endforeach()
if(NOT _oakfield_bench_boost_include_dir AND OAKFIELD_BENCH_BOOST_INCLUDE_DIR)
    set(_oakfield_bench_boost_include_dir "${OAKFIELD_BENCH_BOOST_INCLUDE_DIR}")
endif()

set(_oakfield_bench_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
if(_oakfield_bench_boost_include_dir)
    message(STATUS "Using Boost.Math benchmark headers from ${_oakfield_bench_boost_include_dir}")
    set(CMAKE_REQUIRED_INCLUDES "${_oakfield_bench_boost_include_dir}")
endif()
unset(OAKFIELD_BENCH_HAVE_BOOST_DIGAMMA CACHE)
unset(OAKFIELD_BENCH_HAVE_BOOST_POLYGAMMA CACHE)
unset(OAKFIELD_BENCH_HAVE_BOOST_AIRY CACHE)
unset(OAKFIELD_BENCH_HAVE_BOOST_BESSEL CACHE)
unset(OAKFIELD_BENCH_HAVE_BOOST_ZETA CACHE)
check_include_file_cxx("boost/math/special_functions/digamma.hpp"
    OAKFIELD_BENCH_HAVE_BOOST_DIGAMMA)
check_include_file_cxx("boost/math/special_functions/polygamma.hpp"
    OAKFIELD_BENCH_HAVE_BOOST_POLYGAMMA)
check_include_file_cxx("boost/math/special_functions/airy.hpp"
    OAKFIELD_BENCH_HAVE_BOOST_AIRY)
check_include_file_cxx("boost/math/special_functions/bessel.hpp"
    OAKFIELD_BENCH_HAVE_BOOST_BESSEL)
check_include_file_cxx("boost/math/special_functions/zeta.hpp"
    OAKFIELD_BENCH_HAVE_BOOST_ZETA)
set(CMAKE_REQUIRED_INCLUDES "${_oakfield_bench_saved_required_includes}")
unset(_oakfield_bench_saved_required_includes)
if(OAKFIELD_BENCH_HAVE_BOOST_DIGAMMA
        AND OAKFIELD_BENCH_HAVE_BOOST_POLYGAMMA
        AND OAKFIELD_BENCH_HAVE_BOOST_AIRY
        AND OAKFIELD_BENCH_HAVE_BOOST_BESSEL
        AND OAKFIELD_BENCH_HAVE_BOOST_ZETA)
    set(OAKFIELD_BENCH_HAVE_BOOST_MATH ON)
else()
    set(OAKFIELD_BENCH_HAVE_BOOST_MATH OFF)
endif()

if(PkgConfig_FOUND)
    pkg_check_modules(MPFR QUIET IMPORTED_TARGET mpfr)
    pkg_check_modules(GMP QUIET IMPORTED_TARGET gmp)
endif()

oakfield_add_benchmark(benchmark_core benchmark_core.c)
oakfield_add_benchmark(benchmark_experimental_math benchmark_experimental_math.c)
oakfield_add_benchmark(benchmark_special_functions benchmark_special_functions.cpp)
oakfield_add_benchmark(benchmark_diffusion benchmark_diffusion.c)
oakfield_add_benchmark(benchmark_integrators benchmark_integrators.c)
oakfield_add_benchmark(benchmark_segmented_sieve benchmark_segmented_sieve.c)

if(OAKFIELD_BENCH_HAVE_BOOST_MATH)
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_BOOST_MATH=1)
    if(_oakfield_bench_boost_include_dir)
        target_include_directories(benchmark_special_functions
            PRIVATE "${_oakfield_bench_boost_include_dir}")
    endif()
else()
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_BOOST_MATH=0)
endif()

unset(_oakfield_bench_boost_include_dir)
unset(_oakfield_bench_boost_include_candidates)
unset(_oakfield_bench_boost_candidate)
unset(_oakfield_bench_homebrew_boost_include_dirs)

if(GSL_FOUND)
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_GSL=1)
    target_link_libraries(benchmark_special_functions
        PRIVATE GSL::gsl GSL::gslcblas)
else()
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_GSL=0)
endif()

if(MPFR_FOUND AND GMP_FOUND)
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_MPFR=1)
    target_link_libraries(benchmark_special_functions
        PRIVATE PkgConfig::MPFR PkgConfig::GMP)
else()
    target_compile_definitions(benchmark_special_functions
        PRIVATE OAKFIELD_BENCH_HAVE_MPFR=0)
endif()
