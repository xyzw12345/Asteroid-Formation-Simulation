#include "cuda_n2_backend.h"
#include "../cuda_utils.h"   // For CUDA_CHECK, CUDA_KERNEL_CHECK
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For __global__, __device__
#include <cmath>                      // For sqrt on host (use device math in kernel)
#include <limits>                     // For std::numeric_limits
#include <algorithm>                  // For std::min, std::copy
#include <vector>

// For GPU reduction to find min_dt (if not writing custom kernel)
// You might need to ensure Thrust is findable by CMake if you use it.
// #include <thrust/device_vector.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// For now, let's implement a simpler custom reduction or CPU-side finish.


// --- Kernel for N^2 Acceleration Computation ---
__global__ void compute_accel_n2_kernel(
    const double* __restrict__ d_posX, const double* __restrict__ d_posY, const double* __restrict__ d_posZ,
    const double* __restrict__ d_mass, const bool* __restrict__ d_active,
    double* __restrict__ d_accX, double* __restrict__ d_accY, double* __restrict__ d_accZ,
    int N_capacity, double G, double softening_sq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N_capacity) return; // Thread out of bounds

    // Initialize accelerations for this particle to zero
    // (Alternatively, cudaMemset can be used before kernel launch)
    d_accX[i] = 0.0;
    d_accY[i] = 0.0;
    d_accZ[i] = 0.0;

    if (!d_active[i]) {
        return; // Skip inactive particles
    }

    double pxi = d_posX[i];
    double pyi = d_posY[i];
    double pzi = d_posZ[i];
    
    double sum_ax_i = 0.0;
    double sum_ay_i = 0.0;
    double sum_az_i = 0.0;

    for (int j = 0; j < N_capacity; ++j) {
        if (i == j || !d_active[j]) continue;

        double dx = d_posX[j] - pxi;
        double dy = d_posY[j] - pyi;
        double dz = d_posZ[j] - pzi;

        double r_sq = dx * dx + dy * dy + dz * dz;
        double r_val = sqrt(r_sq); // Use device sqrt

        if (r_val < 1e-9) continue; // Avoid division by zero if particles are at the same spot

        double denominator_factor = max(r_sq, softening_sq) * r_val;
        if (fabs(denominator_factor) < 1e-30) { // Check for very small denominator
            denominator_factor = (denominator_factor < 0) ? -1e-30 : 1e-30;
        }
        
        double force_scalar_component = G * d_mass[j] / denominator_factor;

        sum_ax_i += dx * force_scalar_component;
        sum_ay_i += dy * force_scalar_component;
        sum_az_i += dz * force_scalar_component;
    }
    d_accX[i] = sum_ax_i;
    d_accY[i] = sum_ay_i;
    d_accZ[i] = sum_az_i;
}


// --- Kernel for N^2 Collision Detection ---
// This kernel identifies pairs (i, j) where i < j that are colliding.
// It writes these pairs to global memory buffers d_collision_pairs_p1 and d_collision_pairs_p2.
// d_num_found_pairs is an atomic counter for the number of pairs.
__global__ void detect_collisions_n2_kernel(
    const double* __restrict__ d_posX, const double* __restrict__ d_posY, const double* __restrict__ d_posZ,
    const double* __restrict__ d_radius, const bool* __restrict__ d_active,
    int* __restrict__ d_collision_pairs_p1, int* __restrict__ d_collision_pairs_p2,
    int* __restrict__ d_num_found_pairs, // Atomic counter
    int N_capacity, int max_pairs_buffer_size)
{
    // Each thread handles one particle 'i' and checks against 'j > i'
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N_capacity || !d_active[i]) {
        return;
    }

    double pxi = d_posX[i];
    double pyi = d_posY[i];
    double pzi = d_posZ[i];
    double ri  = d_radius[i];

    for (int j = i + 1; j < N_capacity; ++j) {
        if (!d_active[j]) continue;

        double dx = d_posX[j] - pxi;
        double dy = d_posY[j] - pyi;
        double dz = d_posZ[j] - pzi;

        double dist_sq = dx * dx + dy * dy + dz * dz;
        double sum_radii = ri + d_radius[j];

        if (dist_sq < sum_radii * sum_radii) {
            // Collision detected
            int pair_idx = atomicAdd(d_num_found_pairs, 1); // Get next available slot for the pair
            if (pair_idx < max_pairs_buffer_size) { // Check if buffer has space
                d_collision_pairs_p1[pair_idx] = i; // Store original index i
                d_collision_pairs_p2[pair_idx] = j; // Store original index j
            }
            // If pair_idx >= max_pairs_buffer_size, buffer is full.
            // This collision won't be recorded. A more robust system might have a flag for overflow.
        }
    }
}

// --- Kernel for estimating dt components (simplified version) ---
// Calculates dt_v = r_i / |v_i| and dt_a = sqrt(r_i / |a_i|) for each particle
// and stores the minimum of these into d_dt_components_temp[i].
// A full pairwise check (r_ij / v_rel_ij) on GPU is much more complex to reduce.
__global__ void estimate_dt_components_kernel(
    const double* __restrict__ d_posX, const double* __restrict__ d_posY, const double* __restrict__ d_posZ,
    const double* __restrict__ d_velX, const double* __restrict__ d_velY, const double* __restrict__ d_velZ,
    const double* __restrict__ d_accX, const double* __restrict__ d_accY, const double* __restrict__ d_accZ,
    const bool* __restrict__ d_active,
    double* __restrict__ d_dt_components_temp, // Output buffer for per-particle min_dt
    int N_capacity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N_capacity || !d_active[i]) {
        d_dt_components_temp[i] = 1.0e+20; // Large value if out of bounds, for reduction
        return;
    }

    double min_dt_val = 1.0e+20;

    double v_sq = d_velX[i] * d_velX[i] + d_velY[i] * d_velY[i] + d_velZ[i] * d_velZ[i];
    double a_sq = d_accX[i] * d_accX[i] + d_accY[i] * d_accY[i] + d_accZ[i] * d_accZ[i];
    double characteristic_length = 1.0e+20;

    for (size_t j = 0; j < N_capacity; ++j) { // Could optimize to j = i + 1 if we only care about pairs once
        if (i == j || !d_active[j]) continue;
        double dx = d_posX[j] - d_posX[i];
        double dy = d_posY[j] - d_posY[i];
        double dz = d_posZ[j] - d_posZ[i];
        double dist_sq = dx * dx + dy * dy + dz * dz;
        characteristic_length = min(characteristic_length, sqrt(dist_sq));
        }
    
    if (v_sq > 1e-12) { // Avoid division by zero if particle is stationary
        double dt_v = characteristic_length / std::sqrt(v_sq);
        min_dt_val = min(min_dt_val, dt_v);
    } 

    // Criterion 2: Based on acceleration (time for velocity to change significantly relative to radius)
    // dt_a = safety_factor * sqrt(r_i / |a_i|)
    // This helps with particles starting from rest or in strong gravitational fields.
    if (a_sq > 1e-12) { // Avoid division by zero if no acceleration
        double dt_a = sqrt(characteristic_length / sqrt(a_sq)); // Note sqrt(r/|a|)
        min_dt_val = min(min_dt_val, dt_a);
    }
    
    d_dt_components_temp[i] = min_dt_val;
}


// --- CudaN2Backend Method Implementations ---

CudaN2Backend::CudaN2Backend() {
    // Constructor can initialize any member variables if needed
    // GPU buffer allocation will happen in initialize() or ensure_..._allocated()
}

CudaN2Backend::~CudaN2Backend() {
    // Free GPU memory
    if (d_collision_pairs_p1) CUDA_CHECK(cudaFree(d_collision_pairs_p1));
    if (d_collision_pairs_p2) CUDA_CHECK(cudaFree(d_collision_pairs_p2));
    if (d_num_found_collision_pairs) CUDA_CHECK(cudaFree(d_num_found_collision_pairs));
    if (d_dt_components_temp) CUDA_CHECK(cudaFree(d_dt_components_temp));
}

void CudaN2Backend::initialize(ParticleData* p_data) {
    particles = p_data;
    // Note: ParticleData::allocate_gpu_memory() should be called by the Simulation class
    // after this backend is set and particles are initialized, or if capacity changes.
    // This backend assumes particles->d_* pointers are valid when its methods are called.
}

void CudaN2Backend::ensure_collision_buffers_allocated(size_t required_capacity_pairs) {
    if (required_capacity_pairs > max_gpu_collision_pairs_capacity || 
        !d_collision_pairs_p1 || !d_collision_pairs_p2 || !d_num_found_collision_pairs) {
        
        if (d_collision_pairs_p1) CUDA_CHECK(cudaFree(d_collision_pairs_p1));
        if (d_collision_pairs_p2) CUDA_CHECK(cudaFree(d_collision_pairs_p2));
        if (d_num_found_collision_pairs) CUDA_CHECK(cudaFree(d_num_found_collision_pairs));

        // If required_capacity_pairs is 0 (e.g. no particles), don't allocate.
        if (required_capacity_pairs == 0) {
            max_gpu_collision_pairs_capacity = 0;
            d_collision_pairs_p1 = d_collision_pairs_p2 = d_num_found_collision_pairs = nullptr;
            return;
        }

        // A rough estimate for max pairs: N/2. For very dense scenarios, might need more.
        // Or a fixed reasonably large buffer.
        size_t buffer_size = required_capacity_pairs; // Could be particles->capacity / 2, or a fraction.
                                                      // For simplicity, let's use required_capacity_pairs
                                                      // which should be set based on N.
        if (buffer_size == 0 && particles->capacity > 1) buffer_size = particles->capacity / 2; // Default if 0 passed but particles exist
        if (buffer_size == 0) buffer_size = 1; // Min 1 to avoid zero alloc

        CUDA_CHECK(cudaMalloc(&d_collision_pairs_p1, buffer_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_collision_pairs_p2, buffer_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_found_collision_pairs, sizeof(int)));
        max_gpu_collision_pairs_capacity = buffer_size;
    }
}

void CudaN2Backend::ensure_dt_component_buffer_allocated(size_t required_capacity_particles) {
    if (required_capacity_particles > 0 && (!d_dt_components_temp || required_capacity_particles > particles->capacity /* or some other capacity metric */ )) {
        if (d_dt_components_temp) CUDA_CHECK(cudaFree(d_dt_components_temp));
        CUDA_CHECK(cudaMalloc(&d_dt_components_temp, required_capacity_particles * sizeof(double)));
    } else if (required_capacity_particles == 0 && d_dt_components_temp) {
         CUDA_CHECK(cudaFree(d_dt_components_temp));
         d_dt_components_temp = nullptr;
    }
}


void CudaN2Backend::compute_accelerations() {
    if (!particles || particles->capacity == 0 || !particles->d_posX) { // d_posX check implies GPU mem not ready
        return;
    }

    // Data should already be on GPU (pos, mass, active).
    // Simulation loop is responsible for particles->copy_..._to_gpu() before calling this.

    int N_cap = static_cast<int>(particles->capacity);
    int threads_per_block = 256;
    int blocks_per_grid = (N_cap + threads_per_block - 1) / threads_per_block;

    // No need to cudaMemset accelerations if kernel initializes them.
    // particles->clear_accelerations_gpu(); // Can be called from ParticleData if preferred

    compute_accel_n2_kernel<<<blocks_per_grid, threads_per_block>>>(
        particles->d_posX, particles->d_posY, particles->d_posZ,
        particles->d_mass, particles->d_active,
        particles->d_accX, particles->d_accY, particles->d_accZ,
        N_cap, G_CONST, SOFTENING_EPSILON_SQUARED
    );
    CUDA_KERNEL_CHECK(); // Check for errors after kernel launch
    particles->copy_acc_from_gpu();
}


std::vector<CollisionPair> CudaN2Backend::detect_collisions() {
    std::vector<CollisionPair> host_collision_pairs;
    if (!particles || particles->capacity == 0 || !particles->d_posX) {
        return host_collision_pairs;
    }

    int N_cap = static_cast<int>(particles->capacity);
    // Estimate max possible pairs or use a reasonable buffer size.
    // For N^2, a particle could collide with N-1 others. Worst case (all collide with all) is N*(N-1)/2.
    // A simpler upper bound for buffer is N if each particle is in at most one reported pair for this simplistic model.
    // Let's use N_cap as a pessimistic buffer size for pairs. It can be tuned.
    size_t pairs_buffer_target_size = N_cap > 1 ? N_cap : 1; // Avoid 0
    ensure_collision_buffers_allocated(pairs_buffer_target_size);
    
    if (!d_num_found_collision_pairs) return host_collision_pairs; // Buffer allocation failed

    // Reset atomic counter for number of pairs found on GPU
    CUDA_CHECK(cudaMemset(d_num_found_collision_pairs, 0, sizeof(int)));

    int threads_per_block = 256;
    int blocks_per_grid = (N_cap + threads_per_block - 1) / threads_per_block;

    detect_collisions_n2_kernel<<<blocks_per_grid, threads_per_block>>>(
        particles->d_posX, particles->d_posY, particles->d_posZ,
        particles->d_radius, particles->d_active,
        d_collision_pairs_p1, d_collision_pairs_p2,
        d_num_found_collision_pairs,
        N_cap, static_cast<int>(max_gpu_collision_pairs_capacity)
    );
    CUDA_KERNEL_CHECK();

    // Copy number of found pairs from GPU to CPU
    int num_pairs_found_on_gpu = 0;
    CUDA_CHECK(cudaMemcpy(&num_pairs_found_on_gpu, d_num_found_collision_pairs, sizeof(int), cudaMemcpyDeviceToHost));

    if (num_pairs_found_on_gpu > 0) {
        if (static_cast<size_t>(num_pairs_found_on_gpu) > max_gpu_collision_pairs_capacity) {
            // This indicates buffer overflow on GPU. Some collisions were not recorded.
            // For a robust system, this needs handling (e.g., re-launch with larger buffer, or a flag).
            // For now, just cap it to what we can copy.
            std::cerr << "Warning: GPU collision buffer overflow. Found " << num_pairs_found_on_gpu
                      << " pairs, buffer capacity " << max_gpu_collision_pairs_capacity << std::endl;
            num_pairs_found_on_gpu = static_cast<int>(max_gpu_collision_pairs_capacity);
        }

        std::vector<int> h_p1_indices(num_pairs_found_on_gpu);
        std::vector<int> h_p2_indices(num_pairs_found_on_gpu);

        CUDA_CHECK(cudaMemcpy(h_p1_indices.data(), d_collision_pairs_p1, num_pairs_found_on_gpu * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_p2_indices.data(), d_collision_pairs_p2, num_pairs_found_on_gpu * sizeof(int), cudaMemcpyDeviceToHost));

        host_collision_pairs.reserve(num_pairs_found_on_gpu);
        for (int k = 0; k < num_pairs_found_on_gpu; ++k) {
            host_collision_pairs.push_back({h_p1_indices[k], h_p2_indices[k]});
        }
    }
    return host_collision_pairs;
}


double CudaN2Backend::estimate_min_dt_component(double safety_factor) {
    if (!particles || particles->capacity == 0 || !particles->d_posX) {
        return std::numeric_limits<double>::max(); // Default large dt
    }
    
    int N_cap = static_cast<int>(particles->capacity);
    ensure_dt_component_buffer_allocated(N_cap);
    if (!d_dt_components_temp) return std::numeric_limits<double>::max();

    int threads_per_block = 256;
    int blocks_per_grid = (N_cap + threads_per_block - 1) / threads_per_block;

    estimate_dt_components_kernel<<<blocks_per_grid, threads_per_block>>>(
        particles->d_posX, particles->d_posY, particles->d_posZ,
        particles->d_velX, particles->d_velY, particles->d_velZ,
        particles->d_accX, particles->d_accY, particles->d_accZ,
        particles->d_active,
        d_dt_components_temp, N_cap
    );
    CUDA_KERNEL_CHECK();

    // Manual (simplified) reduction: copy array to host and find min on CPU.
    // Not efficient for large N, but simpler than writing a full GPU reduction kernel.
    std::vector<double> h_dt_components(N_cap);
    CUDA_CHECK(cudaMemcpy(h_dt_components.data(), d_dt_components_temp, N_cap * sizeof(double), cudaMemcpyDeviceToHost));
    
    double min_dt_on_host = std::numeric_limits<double>::max();
    for (int i = 0; i < N_cap; ++i) {
        if (!particles->active[i]) continue;
        min_dt_on_host = std::min(min_dt_on_host, h_dt_components[i]);
    }

    if (min_dt_on_host >= 1.0e+19 && N_cap > 0) { // If all values were very large (e.g., only inactive particles)
        return 1.0; // Default large dt to be clamped by simulation's dt_max
    }

    return min_dt_on_host * safety_factor;
}