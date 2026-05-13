if(NOT DEFINED OAKFIELD_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR "OAKFIELD_PUBLIC_INCLUDE_DIR is required")
endif()

file(GLOB_RECURSE OAKFIELD_PUBLIC_HEADERS
    "${OAKFIELD_PUBLIC_INCLUDE_DIR}/oakfield/*.h")

set(_bad_headers)
foreach(_header IN LISTS OAKFIELD_PUBLIC_HEADERS)
    file(READ "${_header}" _contents)
    if(_contents MATCHES "#[ \t]*include[ \t]*[<\"][^\n\"]*(core/src|backends/src|integrators/src|_internal\\.h|split_internal\\.h|/internal/)")
        list(APPEND _bad_headers "${_header}")
    endif()
endforeach()

if(_bad_headers)
    list(JOIN _bad_headers "\n  " _bad_header_list)
    message(FATAL_ERROR
        "Public headers must not include source-tree or internal headers:\n  ${_bad_header_list}")
endif()
