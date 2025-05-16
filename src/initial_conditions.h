#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "particle_data.h" // For ParticleData
#include <string>

namespace InitialConditions {

    void create_sun_and_asteroid_belt(ParticleData& particles,
                                      int n_asteroids,
                                      double min_orbit_radius_norm, double max_orbit_radius_norm,
                                      double min_mass_norm, double max_mass_norm,
                                      double perturbation_scale);

    // Placeholder for initializing particles from a file
    // void load_from_file(ParticleData& particles, const std::string& filepath);

} // namespace InitialConditions

#endif // INITIAL_CONDITIONS_H