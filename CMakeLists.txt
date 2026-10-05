cmake_minimum_required(VERSION 3.20)

project(thermal
    VERSION 1.0.0
    DESCRIPTION "Thermal conduction solver with GMG preconditioner"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")
endif()

set(COMMON_SOURCES
    thermal.cpp
    solver.cpp
    multigrid.cpp
)

add_executable(thermal ${COMMON_SOURCES})

target_include_directories(thermal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(APPLE)
    find_program(BREW_EXECUTABLE brew)

    if(BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(NOT LIBOMP_PREFIX)
        message(FATAL_ERROR "libomp not found. Install it with: brew install libomp")
    endif()

    message(STATUS "libomp prefix: ${LIBOMP_PREFIX}")

    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)

    target_sources(thermal PRIVATE metal_backend.mm)

    target_compile_options(thermal PRIVATE
        -Xpreprocessor -fopenmp
        -I${LIBOMP_PREFIX}/include
    )

    target_link_libraries(thermal PRIVATE
        ${LIBOMP_PREFIX}/lib/libomp.dylib
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate"
    )

    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

    set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels.metal)
    set(METAL_AIR    ${CMAKE_CURRENT_BINARY_DIR}/kernels.air)
    set(METALLIB     ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${METAL_SOURCE} -o ${METAL_AIR}
        DEPENDS ${METAL_SOURCE}
        COMMENT "Compiling kernels.metal to AIR"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${METALLIB}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR} -o ${METALLIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking AIR into default.metallib"
        VERBATIM
    )

    add_custom_target(metal_shaders ALL DEPENDS ${METALLIB})
    add_dependencies(thermal metal_shaders)

    add_custom_command(TARGET thermal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METALLIB}
            $<TARGET_FILE_DIR:thermal>/default.metallib
        COMMENT "Copying default.metallib next to executable"
        VERBATIM
    )
else()
    find_package(OpenMP REQUIRED)

    target_sources(thermal PRIVATE not_apple.cpp)
    target_link_libraries(thermal PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(thermal PRIVATE /W3)
else()
    target_compile_options(thermal PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

install(TARGETS thermal RUNTIME DESTINATION bin)

message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(APPLE)
    message(STATUS "OpenMP:           manual (libomp at ${LIBOMP_PREFIX})")
    message(STATUS "Metal shaders:    enabled")
else()
    message(STATUS "OpenMP:           find_package")
    message(STATUS "Metal shaders:    disabled")
endif()
