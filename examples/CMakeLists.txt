function(oakfield_add_example name source)
    add_executable(${name} ${source})
    oakfield_suppress_duplicate_library_warnings(${name})
    target_include_directories(${name}
        PRIVATE
            $<TARGET_PROPERTY:simcore,INTERFACE_INCLUDE_DIRECTORIES>)
    if(ARGN)
        target_link_libraries(${name} PRIVATE ${ARGN})
    else()
        target_link_libraries(${name} PRIVATE simcore)
    endif()
endfunction()

oakfield_add_example(example_minimal_field minimal_field.c)
oakfield_add_example(example_cpu_backend_kernel cpu_backend_kernel.c)
oakfield_add_example(example_operator_context operator_context.c simintegrators)
oakfield_add_example(example_integrator_rk4 integrator_rk4.c simintegrators)
oakfield_add_example(example_stochastic_noise stochastic_noise.c simintegrators)
oakfield_add_example(example_special_math special_math.c)
