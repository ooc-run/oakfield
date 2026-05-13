find_package(PkgConfig REQUIRED)

pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

add_executable(oakfield_minimax minimax.cpp)
target_compile_features(oakfield_minimax PRIVATE cxx_std_17)
target_link_libraries(oakfield_minimax PRIVATE PkgConfig::MPFR PkgConfig::GMP)
