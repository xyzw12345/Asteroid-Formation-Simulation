#include "initial_conditions.h"
#include "constants.h" // For G_CONST (SI), AU (SI), M_PI
#include <random>
#include <cmath>
#include <iostream>

namespace InitialConditions {

void create_test_disk_normalized_units(ParticleData& particles,
                                       int n_asteroids,
                                       double min_orbit_radius_norm, double max_orbit_radius_norm,
                                       double min_mass_norm, double max_mass_norm,
                                       double perturbation_scale) {

    if (n_asteroids < 0) {
        std::cerr << "Error [InitialConditions]: Number of asteroids must be non-negative." << std::endl;
        return;
    }
    int num_total_particles_to_create = n_asteroids + 1; // +1 for the Sun

    if (particles.capacity < static_cast<size_t>(num_total_particles_to_create + particles.current_num_particles)) {
        std::cerr << "Error [InitialConditions]: ParticleData capacity (" << particles.capacity
                  << ") is insufficient for " << n_asteroids << " new asteroids (current: "
                  << particles.current_num_particles << "). Call particles.initialize_storage() with larger capacity." << std::endl;
        return;
    }

    int sun_idx = particles.add_particle(0.0, 0.0, 0.0,      // Position (SI)
                                         0.0, 0.0, 0.0,      // Velocity (SI)
                                         1, 4.64e-3);
    
    double asteroid_density = 9.280e6
    std::mt19937 rng(1)
    // std::mt19937 rng(std::random_device{}()); // Use random_device for better seeding, or a fixed seed for reproducibility
    std::uniform_real_distribution<double> dist_uniform_01(0.0, 1.0); // For converting to specific ranges
    std::normal_distribution<double> dist_normal_01(0.0, 1.0);   // Standard normal distribution

    for (int i = 0; i < n_asteroids; ++i) {
        if (particles.current_num_particles >= particles.capacity) break;

        // Position (in normalized units first)
        double r_norm = min_orbit_radius_norm + dist_uniform_01(rng) * (max_orbit_radius_norm - min_orbit_radius_norm);
        double theta_norm = dist_uniform_01(rng) * 2.0 * M_PI;
        
        double x_norm = r_norm * std::cos(theta_norm);
        double y_norm = r_norm * std::sin(theta_norm);
        // z = np.random.normal(0, 0.01 * r)
        double z_norm = dist_normal_01(rng) * (0.01 * r_norm);

        // Velocity (Keplerian in normalized units + perturbation)
        // speed_circ_norm = np.sqrt(G_norm * SUN_MASS_norm / r_norm)
        double speed_circ_norm = std::sqrt(G_norm * sun_mass_norm / r_norm);
        
        double vx_norm = -speed_circ_norm * std::sin(theta_norm);
        double vy_norm = speed_circ_norm * std::cos(theta_norm);
        double vz_norm = 0.0;

        // Add small random velocity component
        vx_norm += dist_normal_01(rng) * perturbation_scale * speed_circ_norm;
        vy_norm += dist_normal_01(rng) * perturbation_scale * speed_circ_norm;
        vz_norm += dist_normal_01(rng) * perturbation_scale * speed_circ_norm * 0.1;

        // Mass (in normalized units) and Radius (calculated using normalized density)
        double mass_norm = min_mass_norm + dist_uniform_01(rng) * (max_mass_norm - min_mass_norm);
        double radius_norm = std::cbrt((3.0 * mass_norm) / (4.0 * M_PI * asteroid_density));
        
        particles.add_particle(x_norm, y_norm, z_norm,
                               vx_norm, vy_norm, vz_norm,
                               mass_norm, radius_norm);
    }

    std::cout << "[InitialConditions] Generated " << particles.current_num_particles
              << " total particles using create_test_disk_normalized_units." << std::endl;
}

} // namespace InitialConditions