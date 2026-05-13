cmake_minimum_required(VERSION 3.20)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.7" CACHE STRING "Minimum macOS deployment target" FORCE)
endif()

project(oakfield
    VERSION 1.0.0
    LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckLinkerFlag)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OAKFIELD_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path to the Oakfield source root")

if(EXISTS "${OAKFIELD_ROOT}/core/src/field.c")
    set(OAKFIELD_SOURCE_DIR "${OAKFIELD_ROOT}")
else()
    message(FATAL_ERROR
        "Could not find Oakfield sources. Expected either "
        "${OAKFIELD_ROOT}/core/src/field.c or "
        "${CMAKE_CURRENT_LIST_DIR}/core/src/field.c.")
endif()

set(OAKFIELD_OUTPUT_ROOT "${CMAKE_BINARY_DIR}")
set(OAKFIELD_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
set(OAKFIELD_PRIVATE_INCLUDE_DIRS
    "${OAKFIELD_SOURCE_DIR}/core/src"
    "${OAKFIELD_SOURCE_DIR}/backends/src"
    "${OAKFIELD_SOURCE_DIR}/integrators/src")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OAKFIELD_OUTPUT_ROOT}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OAKFIELD_OUTPUT_ROOT}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OAKFIELD_OUTPUT_ROOT}/lib")

if(APPLE)
    add_compile_options(
        -Wno-unguarded-availability-new
        -Wno-deprecated-declarations
        -Wno-typedef-redefinition
        -Wno-macro-redefined)
    check_linker_flag(C "LINKER:-no_warn_duplicate_libraries"
        OAKFIELD_LINKER_SUPPORTS_NO_WARN_DUPLICATE_LIBRARIES)
endif()

option(OAKFIELD_ENABLE_FAST_MATH "Enable aggressive floating-point optimizations (-ffast-math)" ON)
option(OAKFIELD_ENABLE_SYMBOLIC_KERNELS "Enable symbolic kernel support" OFF)
option(OAKFIELD_ENABLE_DIAGNOSTICS "Enable diagnostic code paths" OFF)
option(OAKFIELD_ENABLE_ZETA_CORE "Build Zeta/Xi core and operators" ON)
option(OAKFIELD_ENABLE_OPENMP "Enable OpenMP support for CPU backend when available" ON)
option(OAKFIELD_ENABLE_VDSP "Enable Accelerate/vDSP on Apple platforms" ON)
option(OAKFIELD_ENABLE_CUDA "Enable experimental CUDA backend in the build" OFF)
option(OAKFIELD_ENABLE_METAL "Enable experimental Metal backend in the build" OFF)
option(OAKFIELD_ENABLE_EXTENDED_FEATURES "Enable extended feature set" ON)
option(OAKFIELD_BUILD_DOCS "Generate API documentation with Doxygen" OFF)
option(OAKFIELD_BUILD_TESTS "Build Oakfield tests" ON)
option(OAKFIELD_BUILD_BENCHMARKS "Build Oakfield benchmarks" OFF)
option(OAKFIELD_BUILD_EXAMPLES "Build Oakfield example programs" ON)
option(OAKFIELD_BUILD_APPROX_TOOLS "Build developer approximation-generation tools" OFF)

if(DEFINED OAKFIELD_ENABLE_PRO)
    set(OAKFIELD_ENABLE_EXTENDED_FEATURES "${OAKFIELD_ENABLE_PRO}" CACHE BOOL
        "Enable extended feature set" FORCE)
endif()

find_package(Threads REQUIRED)

set(SIM_HAVE_CUDA OFF)
if(OAKFIELD_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET CUDA::cuda_driver OR NOT TARGET CUDA::nvrtc)
        message(FATAL_ERROR
            "OAKFIELD_ENABLE_CUDA=ON requires the CUDA driver and NVRTC libraries. "
            "Set CUDAToolkit_ROOT if CMake cannot find the complete CUDA Toolkit.")
    endif()
    set(SIM_HAVE_CUDA ON)
endif()

set(SIM_HAVE_METAL OFF)
if(OAKFIELD_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR
            "OAKFIELD_ENABLE_METAL=ON requires Apple platforms with the Metal framework.")
    endif()

    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(SIM_HAVE_METAL ON)
endif()

if(OAKFIELD_ENABLE_SYMBOLIC_KERNELS)
    add_compile_definitions(OAKFIELD_ENABLE_SYMBOLIC_KERNELS=1)
else()
    add_compile_definitions(OAKFIELD_ENABLE_SYMBOLIC_KERNELS=0)
endif()

if(OAKFIELD_ENABLE_DIAGNOSTICS)
    add_compile_definitions(OAKFIELD_ENABLE_DIAGNOSTICS=1)
else()
    add_compile_definitions(OAKFIELD_ENABLE_DIAGNOSTICS=0)
endif()

if(OAKFIELD_ENABLE_EXTENDED_FEATURES)
    add_compile_definitions(OAKFIELD_ENABLE_PRO=1)
else()
    add_compile_definitions(OAKFIELD_ENABLE_PRO=0)
endif()

if(OAKFIELD_ENABLE_ZETA_CORE)
    add_compile_definitions(OAKFIELD_ENABLE_ZETA_CORE=1)
else()
    add_compile_definitions(OAKFIELD_ENABLE_ZETA_CORE=0)
endif()

add_compile_definitions(
    _GNU_SOURCE
    OAKFIELD_ENABLE_RTLSDR=0
    OAKFIELD_PROJECT_VERSION_STR="${PROJECT_VERSION}"
    OAKFIELD_MARKETING_VERSION_STR="${PROJECT_VERSION}"
    OAKFIELD_BUILD_NUMBER_STR="oakfield"
    OAKFIELD_BUILD_ID_STR="${PROJECT_VERSION}+oakfield"
    OAKFIELD_GIT_COMMIT_STR="unknown")

set(SIM_FAST_MATH_FLAGS "")
if(OAKFIELD_ENABLE_FAST_MATH)
    list(APPEND SIM_FAST_MATH_FLAGS "-ffast-math" "-fno-math-errno")
endif()

set(SIM_HAVE_OPENMP OFF)
if(OAKFIELD_ENABLE_OPENMP)
    if(APPLE AND CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
        message(STATUS "Disabling OpenMP for AppleClang on macOS.")
    else()
        find_package(OpenMP QUIET)
        if(OpenMP_C_FOUND)
            set(SIM_HAVE_OPENMP ON)
        else()
            message(STATUS "OpenMP not found; CPU backend will be built without OpenMP support.")
        endif()
    endif()
endif()

set(SIM_HAVE_VDSP OFF)
if(APPLE AND OAKFIELD_ENABLE_VDSP)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        set(SIM_HAVE_VDSP ON)
        message(STATUS "vDSP enabled.")
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/EngineSources.cmake")

function(oakfield_make_sources out_var)
    set(_absolute)
    foreach(_source IN LISTS ARGN)
        if(IS_ABSOLUTE "${_source}")
            list(APPEND _absolute "${_source}")
        elseif(_source MATCHES "^/(.+)$")
            list(APPEND _absolute "${OAKFIELD_SOURCE_DIR}/${CMAKE_MATCH_1}")
        else()
            list(APPEND _absolute "${OAKFIELD_ROOT}/${_source}")
        endif()
    endforeach()
    set(${out_var} ${_absolute} PARENT_SCOPE)
endfunction()

set(OAKFIELD_SIMCORE_SOURCES ${OAKFIELD_SIMCORE_BASE_SOURCES})
if(OAKFIELD_ENABLE_ZETA_CORE)
    list(APPEND OAKFIELD_SIMCORE_SOURCES ${OAKFIELD_SIMCORE_ZETA_SOURCES})
endif()
if(SIM_HAVE_CUDA)
    list(APPEND OAKFIELD_SIMCORE_SOURCES ${OAKFIELD_BACKEND_CUDA_SOURCES})
endif()
if(SIM_HAVE_METAL)
    list(APPEND OAKFIELD_SIMCORE_SOURCES ${OAKFIELD_BACKEND_METAL_SOURCES})
endif()

oakfield_make_sources(OAKFIELD_SIMCORE_ABS_SOURCES ${OAKFIELD_SIMCORE_SOURCES})
oakfield_make_sources(OAKFIELD_SPECIAL_MATH_ABS_SOURCES ${OAKFIELD_SPECIAL_MATH_SOURCES})
oakfield_make_sources(OAKFIELD_SIMINTEGRATOR_ABS_SOURCES ${OAKFIELD_SIMINTEGRATOR_SOURCES})
if(SIM_HAVE_CUDA)
    oakfield_make_sources(OAKFIELD_BACKEND_CUDA_ABS_SOURCES ${OAKFIELD_BACKEND_CUDA_SOURCES})
endif()
if(SIM_HAVE_METAL)
    oakfield_make_sources(OAKFIELD_BACKEND_METAL_ABS_SOURCES ${OAKFIELD_BACKEND_METAL_SOURCES})
endif()

if(SIM_HAVE_CUDA)
    set_source_files_properties(
        ${OAKFIELD_BACKEND_CUDA_ABS_SOURCES}
        PROPERTIES
        LANGUAGE CXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_source_files_properties(
            ${OAKFIELD_BACKEND_CUDA_ABS_SOURCES}
            PROPERTIES
            COMPILE_OPTIONS "-x;c++")
    endif()
endif()

if(SIM_HAVE_METAL AND CMAKE_OBJCXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(
        ${OAKFIELD_BACKEND_METAL_ABS_SOURCES}
        PROPERTIES
        COMPILE_OPTIONS "-Wno-nan-infinity-disabled")
endif()

add_library(simcore STATIC ${OAKFIELD_SIMCORE_ABS_SOURCES})
add_library(oakfield::simcore ALIAS simcore)
set_target_properties(simcore PROPERTIES OUTPUT_NAME "simcore")
target_include_directories(simcore
    PUBLIC
        "$<BUILD_INTERFACE:${OAKFIELD_PUBLIC_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        ${OAKFIELD_PRIVATE_INCLUDE_DIRS})
target_link_libraries(simcore PUBLIC Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(simcore PUBLIC m)
endif()

if(SIM_HAVE_VDSP)
    target_compile_definitions(simcore PUBLIC SIM_HAVE_VDSP)
    target_link_libraries(simcore PRIVATE "${ACCELERATE_FRAMEWORK}")
endif()

if(SIM_HAVE_OPENMP)
    target_link_libraries(simcore PRIVATE OpenMP::OpenMP_C)
    add_compile_definitions(SIM_HAVE_OPENMP=1)
endif()

if(SIM_HAVE_CUDA)
    target_compile_definitions(simcore PUBLIC SIM_HAVE_CUDA=1 OAKFIELD_ENABLE_CUDA=1)
    target_link_libraries(simcore PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()

if(SIM_HAVE_METAL)
    target_compile_definitions(simcore PUBLIC SIM_HAVE_METAL=1 OAKFIELD_ENABLE_METAL=1)
    target_link_libraries(simcore PRIVATE "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
endif()

if(OAKFIELD_ENABLE_FAST_MATH)
    target_compile_options(simcore PRIVATE ${SIM_FAST_MATH_FLAGS})
endif()

function(oakfield_suppress_duplicate_library_warnings target)
    if(OAKFIELD_LINKER_SUPPORTS_NO_WARN_DUPLICATE_LIBRARIES)
        target_link_options(${target} PRIVATE "LINKER:-no_warn_duplicate_libraries")
    endif()
endfunction()

set_source_files_properties(
    ${OAKFIELD_SPECIAL_MATH_ABS_SOURCES}
    PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-fno-finite-math-only")

if(OAKFIELD_ENABLE_ZETA_CORE)
    oakfield_make_sources(OAKFIELD_SIMCORE_ZETA_ABS_SOURCES ${OAKFIELD_SIMCORE_ZETA_SOURCES})
    set_source_files_properties(
        ${OAKFIELD_SIMCORE_ZETA_ABS_SOURCES}
        PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-fno-finite-math-only")
endif()

set_source_files_properties(
    "${OAKFIELD_SOURCE_DIR}/core/src/operators/common/warp_safety.c"
    "${OAKFIELD_SOURCE_DIR}/core/src/operators/advection/analytic_warp.c"
    "${OAKFIELD_SOURCE_DIR}/core/src/operators/nonlinear/chaos_map.c"
    "${OAKFIELD_SOURCE_DIR}/core/src/plane_chart.c"
    PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-fno-finite-math-only")

add_library(simintegrators STATIC ${OAKFIELD_SIMINTEGRATOR_ABS_SOURCES})
add_library(oakfield::simintegrators ALIAS simintegrators)
set_target_properties(simintegrators PROPERTIES OUTPUT_NAME "simintegrators")
target_include_directories(simintegrators
    PUBLIC
        "$<BUILD_INTERFACE:${OAKFIELD_PUBLIC_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        ${OAKFIELD_PRIVATE_INCLUDE_DIRS})
target_link_libraries(simintegrators PUBLIC simcore)
# simcore context/state code references integrator registry helpers, while
# integrator implementations call back into simcore context helpers. Declare the
# static archive cycle so one-pass linkers such as GNU ld rescan both libraries.
target_link_libraries(simcore PUBLIC simintegrators)

set(OAKFIELD_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/oakfield"
    CACHE STRING
    "Install directory for Oakfield CMake package files")

install(
    TARGETS simcore simintegrators
    EXPORT OAKFIELDTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(
    DIRECTORY "${OAKFIELD_PUBLIC_INCLUDE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(
    FILES "${CMAKE_CURRENT_LIST_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}")

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/oakfieldConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OAKFIELDConfig.cmake"
    INSTALL_DESTINATION "${OAKFIELD_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OAKFIELDConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/OAKFIELDConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/OAKFIELDConfigVersion.cmake"
    DESTINATION "${OAKFIELD_INSTALL_CMAKEDIR}")

install(
    EXPORT OAKFIELDTargets
    NAMESPACE oakfield::
    FILE OAKFIELDTargets.cmake
    DESTINATION "${OAKFIELD_INSTALL_CMAKEDIR}")

if(OAKFIELD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(OAKFIELD_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(OAKFIELD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(OAKFIELD_BUILD_APPROX_TOOLS)
    add_subdirectory(tools/approx)
endif()

if(OAKFIELD_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(OAKFIELD_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile")
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile.in"
        "${OAKFIELD_DOXYFILE}"
        @ONLY)
    add_custom_target(oakfield_docs
        COMMAND "${DOXYGEN_EXECUTABLE}" "${OAKFIELD_DOXYFILE}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        COMMENT "Generating Oakfield API documentation"
        VERBATIM)
endif()
