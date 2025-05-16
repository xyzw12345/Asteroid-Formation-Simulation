#ifndef SIMULATION_H
#define SIMULATION_H

#include "particle_data.h"
#include "backends/iphysics_backend.h"
#include "integrator.h"
#include "collision_handler.h"
#include <memory>   // For std::unique_ptr
#include <string>
#include <vector>   // For initial condition parameters potentially

// Forward declaration for specific backends if needed for type checking in main/elsewhere
#ifdef USE_CUDA
// class CudaN2Backend; // Example, if specific check is needed outside simulation.cpp
#endif


class Simulation {
public:
    ParticleData particles; // Contains particle arrays and DSU structure
    
    std::unique_ptr<IPhysicsBackend> physics_backend;
    std::unique_ptr<IIntegrator> integrator;

    // Simulation control parameters
    double current_time = 0.0;         // Current simulation time in year / (2 * pi)
    double dt_adaptive = 0.01;         // Current adaptive timestep in seconds
    int step_count = 0;                // Number of simulation steps taken

    // Configuration for adaptive timestepping
    double config_dt_min = 1e-6;       // Minimum allowed timestep
    double config_dt_max = 1e-3;        // Maximum allowed timestep (adjust based on typical orbital periods)
    double config_adaptive_dt_safety_factor = 0.1; // Factor for dt estimation (e.g., CFL-like)
    double config_dt_output = 0.3; // Example: Output every 30 days

    // Performance tracking (optional)
    double time_spent_physics = 0.0;
    double time_spent_integration = 0.0;
    double time_spent_collision = 0.0;
    // ... other timers

    Simulation();
    ~Simulation(); 

    // Initialization methods
    // Particles MUST be initialized (via ParticleData::initialize_storage and then an InitialConditions method)
    // BEFORE calling run_simulation.
    void initialize_simulation_parameters(double dt_min, double dt_max, double safety_factor, double dt_out);
    
    // Setter methods for components (allowing dependency injection)
    void set_physics_backend(std::unique_ptr<IPhysicsBackend> backend);
    void set_integrator(std::unique_ptr<IIntegrator> ig);

    // Main simulation loop functions
    void run_simulation(double total_simulation_time_s, int max_steps = -1); // Runs until total_time or max_steps
    void run_single_step();                                                // Executes one simulation step

private:
    void update_adaptive_timestep();   // Updates dt_adaptive based on current particle state
    void output_snapshot(int step);    // Placeholder for writing simulation state to file/console
    
    bool is_cuda_backend() const;
};

#endif // SIMULATION_H