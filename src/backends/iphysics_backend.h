// src/backends/iphysics_backend.h
#pragma once
#include "../particle_data.h"
#include <vector> // For collision pairs

struct CollisionPair {
    int p1_idx;
    int p2_idx;
};

class IPhysicsBackend {
public:
    virtual ~IPhysicsBackend() = default;
    virtual void initialize(ParticleData* p_data) { particles = p_data; }
    // Computes accelerations and stores them in particles->accX, accY, accZ
    virtual void compute_accelerations() = 0;
    // Detects collisions and returns a list of pairs of indices of colliding particles
    virtual std::vector<CollisionPair> detect_collisions() = 0;
    // For adaptive timestepping: estimates minimum time to collision or significant event
    virtual double estimate_min_dt_component(double safety_factor) = 0;
    virtual std::string get_name() const = 0;
private:
    ParticleData* particles = nullptr;
};