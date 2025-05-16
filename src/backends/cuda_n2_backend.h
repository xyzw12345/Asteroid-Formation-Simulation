#ifndef CUDA_N2_BACKEND_H
#define CUDA_N2_BACKEND_H

#include "iphysics_backend.h"
#include "../particle_data.h"
#include "../constants.h" // For G_CONST, SOFTENING_EPSILON_SQUARED

// Forward declare Thrust types if used for reductions, to avoid including full Thrust headers here.
// namespace thrust { template <typename T> class device_vector; }

class CudaN2Backend : public IPhysicsBackend {
public:
    CudaN2Backend();
    ~CudaN2Backend() override;

    void initialize(ParticleData* p_data) override;
    void compute_accelerations() override;
    std::vector<CollisionPair> detect_collisions() override;
    double estimate_min_dt_component(double safety_factor) override;
    std::string get_name() const override { return "CUDA N^2 Backend"; }

private:
    ParticleData* particles = nullptr;

    // --- GPU Buffers for Collision Detection (example for one approach) ---
    // Option: A large buffer to store pairs found on GPU
    // This size might need to be dynamic or a pessimistic estimate.
    // For N particles, max N/2 pairs in a single step if one particle collides with one other.
    // If a particle can collide with many, could be up to N * (N-1) / 2 (rare).
    // Let's assume a simpler approach for now where we detect flags or limited pairs.
    //
    // A more robust GPU collision pair generation requires significant effort.
    // For this example, detect_collisions might perform some GPU work to identify
    // potential colliders and then finalize pair generation on CPU, or use a
    // simpler GPU kernel that writes to a pre-allocated buffer.

    // Buffer for storing collision pairs found on GPU (indices)
    int* d_collision_pairs_p1 = nullptr;
    int* d_collision_pairs_p2 = nullptr;
    int* d_num_found_collision_pairs = nullptr; // Atomic counter on GPU
    size_t max_gpu_collision_pairs_capacity = 0;

    // Buffer for dt component estimation per particle
    double* d_dt_components_temp = nullptr;


    // Helper to ensure collision pair buffers are allocated
    void ensure_collision_buffers_allocated(size_t required_capacity);
    void ensure_dt_component_buffer_allocated(size_t required_capacity);
};

#endif // CUDA_N2_BACKEND_H