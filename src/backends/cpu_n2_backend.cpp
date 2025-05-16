#include "cpu_n2_backend.h"
#include "constants.h"
#include <cmath>        // For std::sqrt, std::fabs
#include <vector>       // For std::vector
#include <limits>       // For std::numeric_limits
#include <algorithm>    // For std::min
#include <iostream>

void CpuN2Backend::initialize(ParticleData* p_data) {
    particles = p_data;
}

void CpuN2Backend::compute_accelerations() {
    if (!particles || particles->capacity == 0) {
        return;
    }

    // std::cout << "computing accelerations" << std::endl;

    particles->clear_accelerations_cpu(); // Zero out acc vectors on CPU

    const size_t N = particles->capacity; // Iterate up to capacity, check active flag

    // Make const references to particle data for readability and minor potential optimization
    const auto& posX = particles->posX;
    const auto& posY = particles->posY;
    const auto& posZ = particles->posZ;
    const auto& mass = particles->mass;
    const auto& active = particles->active;

    auto& accX = particles->accX; // Modifiable references
    auto& accY = particles->accY;
    auto& accZ = particles->accZ;

    // for (size_t i = 0; i < N; ++i) {
    //     if (!active[i]) continue;
    //     std::cout << "particle: " << i << '\n'
    //         << "position:" << posX[i] << " " << posY[i] << " " << posZ[i] << '\n'
    //         << "velocity:" << particles->velX[i] << " "  << particles->velY[i] << " " << particles->velZ[i] << std::endl;
    // }
    for (size_t i = 0; i < N; ++i) {
        if (!active[i]) continue;

        double pxi = posX[i];
        double pyi = posY[i];
        double pzi = posZ[i];
        
        double sum_ax_i = 0.0;
        double sum_ay_i = 0.0;
        double sum_az_i = 0.0;

        for (size_t j = 0; j < N; ++j) {
            if (i == j || !active[j]) continue; // Skip self or inactive particles

            double dx = posX[j] - pxi;
            double dy = posY[j] - pyi;
            double dz = posZ[j] - pzi;

            double r_sq = dx * dx + dy * dy + dz * dz;
            r_sq = std::max(r_sq, SOFTENING_EPSILON_SQUARED);
            
            // Gravitational softening to prevent extreme forces at close distances
            // F_vec = G * m_i * m_j * (r_vec / (r^2 + eps^2)^(3/2))
            // a_i_vec = G * m_j * (r_vec / (r^2 + eps^2)^(3/2))
            // Or a more common form: a_i_vec = G * m_j * r_vec / ( (r^2)^(3/2) + eps_cubed_equivalent )
            // Simplified: a_i_vec = (G * m_j / (r^3 + softening_term)) * r_vec
            // Let's use the (r^2 + eps^2) form for the denominator of force magnitude, then multiply by unit vector.
            // Force Magnitude = G * m_i * m_j / (r^2 + SOFTENING_EPSILON_SQUARED)
            // Acceleration a_i = Force Magnitude / m_i = G * m_j / (r^2 + SOFTENING_EPSILON_SQUARED)
            // Acceleration vector a_i_vec = (G * m_j / (r^2 + SOFTENING_EPSILON_SQUARED)) * (r_vec / r)
            // a_i_vec = (G * m_j / ((r^2 + SOFTENING_EPSILON_SQUARED) * r)) * r_vec

            double r_val = std::sqrt(r_sq);

            double denominator_factor = r_sq * r_val;
            if (std::fabs(denominator_factor) < 1e-20) { // Avoid division by zero or extreme values
                 denominator_factor = (denominator_factor < 0 ? -1e-20 : 1e-20);
            }
            
            double force_scalar_over_r_component = G_CONST * mass[j] / denominator_factor;

            sum_ax_i += dx * force_scalar_over_r_component;
            sum_ay_i += dy * force_scalar_over_r_component;
            sum_az_i += dz * force_scalar_over_r_component;
        }
        accX[i] = sum_ax_i;
        accY[i] = sum_ay_i;
        accZ[i] = sum_az_i;
        // std::cout << "acceleration:" << accX[i] << ' ' << accY[i] << ' ' << accZ[i] << std::endl;
    }
}

std::vector<CollisionPair> CpuN2Backend::detect_collisions() {
    std::vector<CollisionPair> collision_pairs;
    if (!particles || particles->capacity == 0) {
        return collision_pairs;
    }

    const size_t N = particles->capacity;
    const auto& posX = particles->posX;
    const auto& posY = particles->posY;
    const auto& posZ = particles->posZ;
    const auto& radius = particles->radius;
    const auto& active = particles->active;

    for (size_t i = 0; i < N; ++i) {
        if (!active[i]) continue;

        // Iterate j from i + 1 to avoid duplicate pairs (i,j) and (j,i), and self-collision (i,i)
        for (size_t j = i + 1; j < N; ++j) {
            if (!active[j]) continue;

            double dx = posX[j] - posX[i];
            double dy = posY[j] - posY[i];
            double dz = posZ[j] - posZ[i];

            double dist_sq = dx * dx + dy * dy + dz * dz;
            double sum_radii = radius[i] + radius[j];

            if (dist_sq < sum_radii * sum_radii) {
                // Collision detected
                collision_pairs.push_back({static_cast<int>(i), static_cast<int>(j)});
            }
        }
    }
    if (collision_pairs.size() != 0) {
        std::cout << "detect_collisions" << std::endl;
        for (const CollisionPair& pair : collision_pairs) {
            std::cout << pair << std::endl;
    }
    }
    return collision_pairs;
}

double CpuN2Backend::estimate_min_dt_component(double safety_factor) {
    if (!particles || particles->num_active_particles == 0) { // Use num_active_particles for quick exit
        return std::numeric_limits<double>::max();
    }

    double min_dt_val = std::numeric_limits<double>::max();
    const size_t N = particles->capacity;

    const auto& posX = particles->posX;
    const auto& posY = particles->posY;
    const auto& posZ = particles->posZ;
    const auto& velX = particles->velX;
    const auto& velY = particles->velY;
    const auto& velZ = particles->velZ;
    const auto& accX = particles->accX;
    const auto& accY = particles->accY;
    const auto& accZ = particles->accZ;
    const auto& active = particles->active;


    for (size_t i = 0; i < N; ++i) {
        if (!active[i]) continue;

        double v_sq = velX[i] * velX[i] + velY[i] * velY[i] + velZ[i] * velZ[i];
        double a_sq = accX[i] * accX[i] + accY[i] * accY[i] + accZ[i] * accZ[i];
        double characteristic_length = std::numeric_limits<double>::max();

        for (size_t j = 0; j < N; ++j) { // Could optimize to j = i + 1 if we only care about pairs once
            if (i == j || !active[j]) continue;
            double dx = posX[j] - posX[i];
            double dy = posY[j] - posY[i];
            double dz = posZ[j] - posZ[i];
            double dist_sq = dx * dx + dy * dy + dz * dz;
            characteristic_length = std::min(characteristic_length, std::sqrt(dist_sq));
            }
        
        // std::cout << "Particle number " << i << ": characteristic length: " << characteristic_length << std::endl;
        if (v_sq > 1e-12) { // Avoid division by zero if particle is stationary
            double dt_v = safety_factor * characteristic_length / std::sqrt(v_sq);
            min_dt_val = std::min(min_dt_val, dt_v);
        } 

        // Criterion 2: Based on acceleration (time for velocity to change significantly relative to radius)
        // dt_a = safety_factor * sqrt(r_i / |a_i|)
        // This helps with particles starting from rest or in strong gravitational fields.
        if (a_sq > 1e-12) { // Avoid division by zero if no acceleration
            double dt_a = safety_factor * std::sqrt(characteristic_length / std::sqrt(a_sq)); // Note sqrt(r/|a|)
            min_dt_val = std::min(min_dt_val, dt_a);
        }    
    }
    // std::cout << "estimate_min_dt_component:" << min_dt_val << std::endl;
    return min_dt_val;
}