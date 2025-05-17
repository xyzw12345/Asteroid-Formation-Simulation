# 1. Minimum CMake Version
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # Increased to 3.18 for better CUDA support

# 2. Project Definition
# Enable CUDA language support. CMake will search for nvcc.
project(NBodyAsteroids LANGUAGES CXX CUDA)

# 3. C++ Standard Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. CUDA Configuration (This block executes only if nvcc is found)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler Found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Toolkit Include Dirs (found by CMake): ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") # CMake 3.10+

    # Match C++ standard for CUDA code if possible
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Define a preprocessor macro that C++ and CUDA code can use
    # This lets you write #ifdef USE_CUDA ... #endif in your code.
    add_compile_definitions(USE_CUDA)

    # Specify CUDA Architectures to compile for.
    # Adjust these based on your target GPU(s) and desired compatibility.
    # Examples:
    #   52 (Maxwell), 61 (Pascal), 70 (Volta), 75 (Turing),
    #   80 (Ampere GA100), 86 (Ampere GA10x), 89 (Ada), 90 (Hopper)
    # 'NATIVE' compiles for the architecture of the build machine's GPU (if detected).
    set(CMAKE_CUDA_ARCHITECTURES 75 86 NATIVE CACHE STRING "CUDA Architectures (e.g., 75;86;NATIVE)")
    message(STATUS "Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

else()
    message(WARNING "CUDA Compiler (nvcc) not found. CUDA-specific features will be disabled.")
    # USE_CUDA will NOT be defined if this branch is taken, so CUDA code will be excluded.
endif()

# 5. Define Source Files
set(PROJECT_SOURCES
    src/main.cpp
    src/particle_data.cpp
    src/initial_conditions.cpp
    src/simulation.cpp
    src/integrator.cpp
    # CPU Backends
    src/backends/cpu_n2_backend.cpp
)

# Conditionally add .cu (CUDA) source files
if(CMAKE_CUDA_COMPILER) # Only if nvcc is found and CUDA language is enabled
    list(APPEND PROJECT_SOURCES
        src/backends/cuda_n2_backend.cu   # Your CUDA N2 backend implementation
    )
endif()

# 6. Add Executable Target
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# 7. Specify Include Directories
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src" # For your project's headers
)

# If CUDA is enabled, CXX files might need to see CUDA headers (e.g. for cudaError_t via cuda_utils.h)
# find_package(CUDAToolkit) provides CUDAToolkit_INCLUDE_DIRS (CMake 3.17+)
# For older CMake, CUDA_INCLUDE_DIRS is often set when CUDA language is enabled.
if(CMAKE_CUDA_COMPILER)
    # If CUDAToolkit_INCLUDE_DIRS is set by find_package(CUDAToolkit), use it.
    # Otherwise, CUDA_INCLUDE_DIRS is usually available when LANGUAGES CUDA is used.
    if(NOT CUDAToolkit_INCLUDE_DIRS AND CUDA_INCLUDE_DIRS)
        target_include_directories(${PROJECT_NAME} INTERFACE ${CUDA_INCLUDE_DIRS})
        message(STATUS "Adding CUDA_INCLUDE_DIRS to CXX compile: ${CUDA_INCLUDE_DIRS}")
    elseif(CUDAToolkit_INCLUDE_DIRS)
         target_include_directories(${PROJECT_NAME} INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
         message(STATUS "Adding CUDAToolkit_INCLUDE_DIRS to CXX compile: ${CUDAToolkit_INCLUDE_DIRS}")
    endif()
    # Note: Using INTERFACE here ensures that if other C++ files include headers
    # from this project that themselves include CUDA headers, they get the path.
    # For just this executable, PRIVATE might also work if only .cu files directly include cuda_runtime.
endif()


# 8. Link Libraries
# Math library (for sqrt, sin, cos, etc.)
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

# CUDA Runtime Library
# When "LANGUAGES CUDA" is used, CMake typically handles linking the CUDA runtime library (cudart)
# automatically for targets that include .cu files.
# If you get linker errors for CUDA runtime functions, you might need to add it explicitly:
# if(CMAKE_CUDA_COMPILER)
#   target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart) # Modern CMake (3.10+)
# endif()


# 9. Compiler Flags
if(MSVC) # Flags specific to Microsoft Visual C++ compiler (cl.exe)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4          # Warning level
        /EHsc        # C++ exception handling model
        # CMake adds /MDd /Zi /Od for Debug and /MD /O2 /DNDEBUG for Release by default.
        # Add more specific flags if needed:
        $<$<CONFIG:Debug>:/MP>      # Example: Enable multi-processor C++ compilation for Debug
        $<$<CONFIG:Release>:/O2 /MP>
    )
    if(CMAKE_CUDA_COMPILER) # Flags for nvcc when host compiler is MSVC
        target_compile_options(${PROJECT_NAME} PRIVATE
            # Flags for CUDA device code compilation by nvcc
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G>>      # Device debug info
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3>>  # Device optimization

            # Flags passed by nvcc to MSVC (the host compiler) for host code in .cu files
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/W3,/EHsc"> # Basic flags for host compiler
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-Xcompiler="/Zi,/Od,/MDd">>
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-Xcompiler="/O2,/MD">>
        )
    endif()
else() # Flags for GCC, Clang, and other compatible compilers
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -pedantic
        $<$<CONFIG:Debug>:-g -pg -O3>
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
    )
    if(CMAKE_CUDA_COMPILER) # Flags for nvcc when host compiler is GCC/Clang
        target_compile_options(${PROJECT_NAME} PRIVATE

            $<$<COMPILE_LANGUAGE:CUDA>:-MD> # For dependency generation
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O3 --use_fast_math -ftz=true>> # Device debug and basic opt
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3 --use_fast_math -ftz=true>> # Max device optimization

            # Flags passed by nvcc to GCC/Clang (the host compiler)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-Xcompiler=-O3;-Xcompiler=-DNDEBUG;-Xcompiler=-march=native;-Xcompiler=-ffast-math>>
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-Xcompiler=-O3;-Xcompiler=-DNDEBUG;-Xcompiler=-march=native;-Xcompiler=-ffast-math>>
        )
    endif()
endif()


# 10. Output Information (Optional, but helpful for users)
message(STATUS "-------------------------------------------------------")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA enabled: YES")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA enabled: NO")
endif()
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------------------------------------")