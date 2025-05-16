#include "simulation.h"
#include "initial_conditions.h" // For particle generation
#include "particle_data.h"      // For ParticleData itself
#include "constants.h"          // For AU, G_CONST, M_PI etc.

// Backend headers for selection
#include "backends/cpu_n2_backend.h"
// #include "backends/cpu_spatial_hash_backend.h"
#ifdef USE_CUDA
// #include "backends/cuda_n2_backend.h"
#endif

#include <iostream>
#include <string>
#include <numbers>
#include <memory>   // For std::make_unique
#include <stdexcept> // For std::stoi, std::stod for robust parsing
#include <cmath>     // For std::pow, std::sqrt, M_PI

// Helper function to parse arguments (basic)
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [num_asteroids] [backend_choice] [sim_duration_years]\n"
              << "  num_asteroids (int): Number of asteroids to generate (e.g., 1000). Default: 100.\n"
              << "  backend_choice (str): 'cpu_n2', 'cuda_n2', 'cpu_spatial_hash'. Default: 'cpu_n2'.\n"
              << "  sim_duration_years (float): Total simulation duration in years (e.g., 0.1). Default: 0.01 years.\n"
              << std::endl;
}


int main(int argc, char** argv) {
    // --- Default Simulation Parameters ---
    int n_asteroids = 100;                 // Default number of asteroids
    std::string backend_choice_str = "cpu_n2"; // Default backend
    double sim_duration_years = 5;      // Default simulation duration in years

    // --- Parse Command-Line Arguments (Basic) ---
    if (argc >= 2) {
        try {
            n_asteroids = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing num_asteroids: " << e.what() << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (argc >= 3) {
        backend_choice_str = argv[2];
    }
    if (argc >= 4) {
        try {
            sim_duration_years = std::stod(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing sim_duration_years: " << e.what() << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
     if (argc > 4) {
        std::cerr << "Too many arguments." << std::endl;
        print_usage(argv[0]);
        return 1;
    }


    std::cout << "--- N-Body Asteroid Formation Simulation ---" << std::endl;
    std::cout << "Selected number of asteroids: " << n_asteroids << std::endl;
    std::cout << "Selected backend: " << backend_choice_str << std::endl;
    std::cout << "Selected simulation duration: " << sim_duration_years << " years." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;


    // --- Create Simulation Object ---
    Simulation sim;

    // --- Initialize ParticleData Storage ---
    // Make capacity a bit larger than num_asteroids + 1 (for Sun) to allow for some buffer
    // or if other particles might be added dynamically (not in current scope).
    int total_particles_capacity = n_asteroids + 1 + 50; // +1 for Sun, +50 buffer
    sim.particles.initialize_storage(total_particles_capacity);
    std::cout << "ParticleData storage initialized with capacity: " << total_particles_capacity << std::endl;

    // --- Populate Particles (using normalized units as in Python example) ---
    // Parameters for create_test_disk_normalized_units
    double min_orbit_radius_norm = 0.95;  
    double max_orbit_radius_norm = 1.05; 
    double min_mass_norm = 1e-5;      
    double max_mass_norm = 3e-5;      
    double perturbation_scale_frac = 0.05; // Velocity perturbation as fraction of Keplerian speed

    InitialConditions::create_test_disk_normalized_units(sim.particles, n_asteroids,
                                                        min_orbit_radius_norm, max_orbit_radius_norm,
                                                        min_mass_norm, max_mass_norm,
                                                        perturbation_scale_frac);

    // --- Initialize Simulation Run Parameters (Timestepping, Output - in SI units) ---
    // These dt values are in SI seconds.
    // A typical time unit in the G=1 system is 1 year / (2*pi).
    // (1 year / (2pi)) ~= 5.02e6 seconds.
    // Let's choose dt based on a fraction of this, or typical orbital periods.
    // For inner belt (1 AU), period is 1 year. dt_max could be 1/100th of that.
    double typical_time_unit_norm_sec = std::sqrt(std::pow(AU, 3) / (G_CONST * 1.989e30)); // Approx 5e6 s
    
    double dt_min_si = typical_time_unit_norm_sec / 10000.0; // e.g., ~500 seconds
    double dt_max_si = typical_time_unit_norm_sec / 50.0;   // e.g., ~1e5 seconds (little over 1 day)
    double safety_factor_dt = 0.05; // For adaptive timestep calculation
    double output_interval_si = typical_time_unit_norm_sec * 0.5 * (2.0 * M_PI) ; // Output approx every 0.5 normalized time units (0.5 * 1_year_norm)
                                                                                 // This is 0.5 year_SI
    
    sim.initialize_simulation_parameters(1e-6, 1e-3, 0.05, 0.3);

    // --- Set Physics Backend ---
    if (backend_choice_str == "cpu_n2") {
        sim.set_physics_backend(std::make_unique<CpuN2Backend>());
    } 
//     else if (backend_choice_str == "cpu_spatial_hash") {
//         sim.set_physics_backend(std::make_unique<CpuSpatialHashBackend>());
//     } 
// #ifdef USE_CUDA
//     else if (backend_choice_str == "cuda_n2") {
//         sim.set_physics_backend(std::make_unique<CudaN2Backend>());
//     }
// #endif
    else {
        std::cerr << "Warning: Unknown backend choice '" << backend_choice_str 
                  << "'. Defaulting to 'cpu_n2'." << std::endl;
        sim.set_physics_backend(std::make_unique<CpuN2Backend>());
    }
    // Integrator is already defaulted to LeapfrogKDK in Simulation constructor.

    // --- Run Simulation ---
    // Convert desired simulation duration from years (SI) to seconds (SI)
    double total_sim_time_seconds = sim_duration_years * 2 * std::numbers::pi;
    int max_steps = -1; // No step limit, run until time limit. Or set a value e.g., 10000.

    try {
        sim.run_simulation(total_sim_time_seconds_si, max_steps);
    } catch (const std::exception& e) {
        std::cerr << "\n!!! Simulation run failed with an exception: " << e.what() << std::endl;
        // Perform any necessary cleanup if resources aren't RAII managed by sim destructor
        return 1;
    } catch (...) {
        std::cerr << "\n!!! Simulation run failed with an unknown exception." << std::endl;
        return 1;
    }

    std::cout << "\nMain function finished." << std::endl;
    return 0;
}