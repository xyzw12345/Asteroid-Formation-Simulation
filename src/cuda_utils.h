#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// This header should only be included by .cu files or files compiled with NVCC,
// or it should guard CUDA-specific code with #ifdef __CUDACC__
// However, for macros like CUDA_CHECK, they can be defined to do something
// useful (like printing an error) even if CUDA is not fully available,
// or simply be no-ops.

#include <cstdio>  // For fprintf, stderr
#include <cstdlib> // For exit

// Only define CUDA-specific includes and functions if USE_CUDA is defined
// (which should be set by your CMakeLists.txt if CUDA is found and enabled)
#ifdef USE_CUDA
#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString, etc.

// Macro to check CUDA call errors
// This version prints the error and exits.
// For a library, you might prefer throwing an exception or returning an error code.
#define CUDA_CHECK(err) __cuda_check_error(err, __FILE__, __LINE__)

// Inline helper function for CUDA_CHECK
// This function should ideally be in a .cu file if it's more complex
// or if you want to avoid including cuda_runtime.h in non-.cu files that might include cuda_utils.h.
// However, for a simple error checker, an inline function in the header is common.
inline void __cuda_check_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
        // For critical errors, you might want to exit or throw
        exit(EXIT_FAILURE);
        // Or: throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + " at " + file + ":" + std::to_string(line));
    }
}

// Optional: A macro to check for errors after kernel launches
// cudaGetLastError() is often used for this purpose.
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())


// Optional: Helper function to print GPU device info (could be in .cu)
// void printCudaDeviceInfo(); // Declaration, definition in .cu

#else // If USE_CUDA is not defined

// Define CUDA_CHECK and CUDA_KERNEL_CHECK as no-ops or error messages if CUDA is not used
#define CUDA_CHECK(err) \
    do { \
        if (err != 0 /*cudaSuccess equivalent for a stub*/) { \
            fprintf(stderr, "CUDA API call made but CUDA is not enabled/available (Error code: %d) at %s:%d\n", static_cast<int>(err), __FILE__, __LINE__); \
            /* exit(EXIT_FAILURE); // Or handle differently */ \
        } \
    } while (0)

#endif // USE_CUDA

#endif // CUDA_UTILS_H