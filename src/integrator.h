// src/integrator.h (Revised)
#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "particle_data.h"
#include "backends/iphysics_backend.h" // Include the physics backend interface
#include <functional> // If we were to use std::function for callbacks

// Abstract base class for integrators
class IIntegrator {
public:
    virtual ~IIntegrator() = default;

    // Performs one full integration step of duration dt.
    // The integrator is responsible for calling the physics_backend to compute
    // accelerations as needed during its internal substeps.
    // It will update particles.posX, posY, posZ, velX, velY, velZ.
    // It should also ensure that particles.accX, accY, accZ are updated to a(t+dt)
    // by the end of the step, typically by making a final call to the physics_backend
    // if its internal logic doesn't already leave them in that state.
    virtual void step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) = 0;

    // Optional: Some integrators might need an initial setup if v and a are not aligned as expected.
    // For example, a basic Leapfrog might need an initial half-kick if v(0) and a(0) are given.
    // KDK with a(0) available is usually fine from the start.
    // virtual void initialize_step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) {
        // Default implementation: do nothing.
        // Can be overridden if an integrator needs a special first half-step.
        // For KDK, if a(0) is already computed and stored in particles.acc before the first 'step' call,
        // this might not be strictly necessary.
    // }
};

// Kick-Drift-Kick Leapfrog Integrator
class LeapfrogKDKIntegrator : public IIntegrator {
public:
    LeapfrogKDKIntegrator() = default;

    void step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) override;

    // KDK expects a(t) to be available at the start of `step`.
    // The `step` method will then compute a(t+dt) and leave it in particles.acc.
};


// Example: Euler Integrator (simple, not for production N-body, but demonstrates interface)
class EulerIntegrator : public IIntegrator {
public:
    EulerIntegrator() = default;
    void step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) override;
};


#endif // INTEGRATOR_H