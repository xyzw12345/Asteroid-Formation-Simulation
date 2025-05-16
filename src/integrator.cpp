#include "integrator.h"
#include "particle_data.h"        // Included via integrator.h
#include "backends/iphysics_backend.h" // Included via integrator.h

// --- LeapfrogKDKIntegrator Implementation ---
void LeapfrogKDKIntegrator::step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) {
    // Assumes:
    // - particles.accX/Y/Z currently hold a(t).
    // - particles.posX/Y/Z hold x(t).
    // - particles.velX/Y/Z hold v(t).

    const size_t N = particles.capacity;

    // 1. First Kick: v(t + dt/2) = v(t) + a(t) * dt/2
    for (size_t i = 0; i < N; ++i) {
        if (!particles.active[i]) continue;

        particles.velX[i] += particles.accX[i] * dt * 0.5;
        particles.velY[i] += particles.accY[i] * dt * 0.5;
        particles.velZ[i] += particles.accZ[i] * dt * 0.5;
    }
    // Now particles.velX/Y/Z hold v(t + dt/2)

    // 2. Drift: x(t + dt) = x(t) + v(t + dt/2) * dt
    for (size_t i = 0; i < N; ++i) {
        if (!particles.active[i]) continue;

        particles.posX[i] += particles.velX[i] * dt;
        particles.posY[i] += particles.velY[i] * dt;
        particles.posZ[i] += particles.velZ[i] * dt;
    }
    // Now particles.posX/Y/Z hold x(t + dt)

    // 3. Compute new accelerations a(t + dt) based on new positions x(t + dt)
    // This is where the integrator calls the physics backend.
    // (If GPU backend) Data transfer logic (CPU positions -> GPU, GPU accelerations -> CPU)
    // would be handled by the physics_backend implementation or around its call.
    // For simplicity, assume physics_backend handles its own GPU sync if needed, or
    // the simulation loop does it before/after this step call.
    // Let's assume for now the physics_backend->compute_accelerations() updates
    // particles.accX/Y/Z directly (or its GPU counterparts, which are then synced).
    
    // Simulation loop must ensure data is on the correct device for physics_backend
    // e.g., if physics_backend is CUDA, new positions must be on GPU.
    // For now, let's assume the physics_backend works with what's in `particles` object,
    // and the simulation orchestrates H<->D transfers if backend is GPU.
    // A more tightly coupled design could have the physics_backend take GPU pointers.
    // The current IPhysicsBackend interface takes ParticleData*, so it can decide.

    // The call sequence in Simulation::run_step would be:
    // IF CUDA backend:
    //   particles.copy_pos_vel_mass_radius_active_id_to_gpu(); // Ensure x(t+dt) and v(t+dt/2) are on GPU
    physics_backend->compute_accelerations(); // This will update particles.accX/Y/Z (or d_accX/Y/Z)
    // IF CUDA backend:
    //   particles.copy_acc_from_gpu(); // Ensure a(t+dt) is on CPU

    // Now particles.accX/Y/Z (CPU side) hold a(t+dt)

    // 4. Second Kick: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    for (size_t i = 0; i < N; ++i) {
        if (!particles.active[i]) continue;

        particles.velX[i] += particles.accX[i] * dt * 0.5;
        particles.velY[i] += particles.accY[i] * dt * 0.5;
        particles.velZ[i] += particles.accZ[i] * dt * 0.5;
    }
    // Now:
    // - particles.posX/Y/Z hold x(t + dt)
    // - particles.velX/Y/Z hold v(t + dt)
    // - particles.accX/Y/Z hold a(t + dt) -> ready for the next step.
}


// --- EulerIntegrator Implementation ---
void EulerIntegrator::step(ParticleData& particles, double dt, IPhysicsBackend* physics_backend) {
    // Assumes:
    // - particles.accX/Y/Z currently hold a(t).
    // - particles.posX/Y/Z hold x(t).
    // - particles.velX/Y/Z hold v(t).

    const size_t N = particles.capacity;

    // 1. Update position: x(t + dt) = x(t) + v(t) * dt
    for (size_t i = 0; i < N; ++i) {
        if (!particles.active[i]) continue;

        particles.posX[i] += particles.velX[i] * dt;
        particles.posY[i] += particles.velY[i] * dt;
        particles.posZ[i] += particles.velZ[i] * dt;
    }

    // 2. Update velocity: v(t + dt) = v(t) + a(t) * dt
    //    Note: Euler uses a(t) for both position and velocity updates for the step t to t+dt.
    for (size_t i = 0; i < N; ++i) {
        if (!particles.active[i]) continue;

        particles.velX[i] += particles.accX[i] * dt;
        particles.velY[i] += particles.accY[i] * dt;
        particles.velZ[i] += particles.accZ[i] * dt;
    }

    // 3. Compute new accelerations a(t + dt) based on new positions x(t + dt) and new velocities v(t+dt)
    //    (though for gravity, acceleration usually only depends on position).
    //    This a(t+dt) will be used for the *next* step.
    // IF CUDA backend:
    //   particles.copy_pos_vel_mass_radius_active_id_to_gpu(); // x(t+dt), v(t+dt)
    physics_backend->compute_accelerations();
    // IF CUDA backend:
    //   particles.copy_acc_from_gpu();

    // Now:
    // - particles.posX/Y/Z hold x(t + dt)
    // - particles.velX/Y/Z hold v(t + dt)
    // - particles.accX/Y/Z hold a(t + dt) -> ready for the next step.
}