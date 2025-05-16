if(WIN32)
    # Try to find the correct cl.exe for the host compiler
    # This path should match what works with -ccbin.
    # You MUST adjust this path to your actual MSVC installation.
    set(MSVC_CL_EXE_PATH "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
    # Or, if you have the full Visual Studio Community/Professional/Enterprise:
    # set(MSVC_CL_EXE_PATH "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")


    if(EXISTS "${MSVC_CL_EXE_PATH}")
        set(CMAKE_CUDA_HOST_COMPILER "${MSVC_CL_EXE_PATH}" CACHE FILEPATH "Host compiler for CUDA")
        message(STATUS "CMAKE_CUDA_HOST_COMPILER explicitly set to: ${CMAKE_CUDA_HOST_COMPILER}")
    else()
        message(WARNING "Specified MSVC_CL_EXE_PATH not found: ${MSVC_CL_EXE_PATH}. CUDA host compiler not explicitly set.")
    endif()
endif()

# Minimum CMake version required
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # Using FATAL_ERROR ensures CMake stops if version is too low

# Project name and languages. CXX for C++, CUDA for NVIDIA CUDA.
# Enabling CUDA language here handles finding the CUDA toolkit and nvcc compiler.
project(NBodyAsteroids LANGUAGES CXX CUDA)

# --- Standard C++ Configuration ---
# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True) # Enforce the standard
set(CMAKE_CXX_EXTENSIONS OFF)         # Prefer not to use compiler-specific extensions by default

# --- CUDA Configuration ---
# The "LANGUAGES CUDA" already does a lot. These are further refinements.
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler Found: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD}) # Match C++ standard for CUDA if possible
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Define a preprocessor macro that C++ and CUDA code can use to check if CUDA is active
    add_compile_definitions(USE_CUDA) # This will define USE_CUDA for all source files

    # Specify CUDA Architectures to compile for (Virtual and Real)
    # Example: Compile for Volta (sm_70), Turing (sm_75), Ampere (sm_80, sm_86), and the native architecture of the build machine.
    # Adjust these based on your target GPUs.
    # Using "Native" helps if you build on the machine you run on.
    # For broader distribution, include common architectures.
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 NATIVE CACHE STRING "CUDA Architectures (e.g., 70;75;86;NATIVE)")
    message(STATUS "Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Optional: Add specific CUDA compiler flags if needed
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math") # Example: use with caution

else()
    message(WARNING "CUDA Compiler (nvcc) not found. CUDA-specific features will be disabled.")
    # USE_CUDA will NOT be defined if this branch is taken.
endif()

# --- Source Files ---
# List all your .cpp source files
set(PROJECT_SOURCES
    src/main.cpp
    src/particle_data.cpp
    src/initial_conditions.cpp
    src/simulation.cpp
    src/integrator.cpp
    src/backends/cpu_n2_backend.cpp
    # src/backends/cpu_spatial_hash_backend.cpp
)

# Conditionally add .cu (CUDA) source files
if(CMAKE_CUDA_COMPILER) # Only if nvcc is found
    list(APPEND PROJECT_SOURCES
        # src/backends/cuda_n2_backend.cu
    )
endif()

# --- Executable Target ---
# Create the executable from the source files
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# --- Include Directories ---
# Tell CMake where to find header files (e.g., src/ for your project headers)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src # Allows #include "particle_data.h"
)
# If you had a third_party/include directory:
# target_include_directories(${PROJECT_NAME} PUBLIC third_party/include)


# --- Linking Libraries ---
# CUDA Runtime Library: Modern CMake with "LANGUAGES CUDA" often handles this automatically.
# If you needed to link it explicitly (older CMake or specific cases):
# if(CMAKE_CUDA_COMPILER)
#   target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart) # Modern CMake way
# endif()

# Math library (for sqrt, sin, cos, etc.)
# On Linux/macOS, you often need to link m (libm)
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

# --- Compiler Flags ---
# Set general compiler flags for C++ and CUDA (can be different)
# These are examples; adjust as needed.

# Common flags for all configurations (Debug, Release, etc.)
# target_compile_options(${PROJECT_NAME} PRIVATE
#     $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic> # C++ specific warnings
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="-Wall,-Wextra"> # Pass warnings to host compiler for CUDA
# )

# Configuration-specific flags (e.g., optimization for Release, debug info for Debug)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:-g -O0>> # Debug flags for C++
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G -O0>> # Debug flags for CUDA (-G enables device debug)

    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>> # Release flags for C++
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>> # Release flags for CUDA
)
# For MSVC (Windows), flags are different, e.g., /Zi for debug, /O2 or /Ox for release.
# CMake often handles these well with CMAKE_BUILD_TYPE, but you can add specific ones.
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Release>:/O2>
        # Common MSVC flags
        /W3 # Warning level 3
    )
else() # For GCC/Clang
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -pedantic # Common warnings
    )
endif()


# --- Output Directory (Optional) ---
# Control where the final executable is placed.
# Default is usually <build_directory>/${PROJECT_NAME}
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# --- Installation (Optional) ---
# If you plan to install the project
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)


# --- Testing with CTest (Optional) ---
# enable_testing()
# add_test(NAME MySimulationTest COMMAND ${PROJECT_NAME} <args_for_test_run>)
# set_tests_properties(MySimulationTest PROPERTIES PASS_REGULAR_EXPRESSION "Simulation finished")


# --- Print Summary ---
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}") # Debug, Release, RelWithDebInfo, MinSizeRel
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: Disabled")
endif()