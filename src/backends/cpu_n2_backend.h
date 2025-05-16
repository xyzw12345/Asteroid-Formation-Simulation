#ifndef CPU_N2_BACKEND_H
#define CPU_N2_BACKEND_H

#include "iphysics_backend.h"
#include "../particle_data.h" // For ParticleData itself
#include "../constants.h"    // For G, SOFTENING_EPSILON_SQUARED

class CpuN2Backend : public IPhysicsBackend {
public:
    CpuN2Backend() = default;
    ~CpuN2Backend() override = default;

    // Initializes the backend with a pointer to the particle data.
    void initialize(ParticleData* p_data) override;

    // Computes gravitational accelerations for all active particles using N^2 direct summation.
    // Results are stored in particles->accX, accY, accZ.
    void compute_accelerations() override;

    // Detects collisions between all pairs of active particles using N^2 check.
    // Returns a vector of CollisionPair, where each pair contains the array indices
    // of the two colliding particles.
    std::vector<CollisionPair> detect_collisions() override;

    // Estimates a component for the adaptive timestep calculation.
    // This involves checking:
    // 1. Time for a particle to travel its own radius (r_i / v_i).
    // 2. Time based on acceleration (sqrt(r_i / a_i)).
    // 3. (Optional but good for slingshot) Estimated time to closest approach for all pairs (r_ij / v_rel_ij).
    // Returns the minimum of these estimated times multiplied by a safety factor.
    double estimate_min_dt_component(double safety_factor) override;

    std::string get_name() const override { return "CPU N^2 Backend"; }

private:
    ParticleData* particles = nullptr; // Pointer to the particle data
};

#endif // CPU_N2_BACKEND_H