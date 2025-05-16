#include "simulation.h"
#include "constants.h"      // For G_CONST, AU, etc.
#include "initial_conditions.h" // For particle setup
#include "backends/cpu_n2_backend.h" 
#include "integrator.h"     
#ifdef USE_CUDA
// #include "backends/cuda_n2_backend.h" 
#endif

#include <iostream>
#include <algorithm>    // For std::min/max
#include <chrono>       
#include <iomanip>      

Simulation::Simulation() {
    // ParticleData is default constructed. It needs initialize_storage called.
    // Physics backend and integrator will be fully initialized once particles are ready and backend is set.
    set_physics_backend(std::make_unique<CpuN2Backend>()); // Set a default
    set_integrator(std::make_unique<LeapfrogKDKIntegrator>()); // Set a default
    std::cout << "Simulation object created." << std::endl;
}

Simulation::~Simulation() {
    std::cout << "Simulation object destroying..." << std::endl;
    if (is_cuda_backend() && particles.capacity > 0) { // If particles were ever used with CUDA
        std::cout << "Freeing GPU memory via ParticleData from Simulation destructor." << std::endl;
        particles.free_gpu_memory(); 
    }
    std::cout << "Simulation object destroyed." << std::endl;
}

void Simulation::initialize_simulation_parameters(double dt_min, double dt_max, double safety_factor, double dt_out) {
    config_dt_min = dt_min;
    config_dt_max = dt_max;
    config_adaptive_dt_safety_factor = safety_factor;
    config_dt_output = dt_out;
    dt_adaptive = config_dt_max; 
    std::cout << "Simulation parameters initialized: dt_min=" << config_dt_min
              << "s, dt_max=" << config_dt_max << "s, safety_factor="
              << config_adaptive_dt_safety_factor << std::endl;
}


void Simulation::set_physics_backend(std::unique_ptr<IPhysicsBackend> backend) {
    physics_backend = std::move(backend);
    if (physics_backend && particles.capacity > 0) { 
        physics_backend->initialize(&particles); // Initialize backend with existing particle data
        std::cout << "Physics backend set to: " << physics_backend->get_name() << std::endl;
    } else if (physics_backend) {
        std::cout << "Physics backend set to: " << physics_backend->get_name() 
                  << " (particles capacity not yet set or zero; backend will be initialized later if capacity changes)" << std::endl;
    }
}

void Simulation::set_integrator(std::unique_ptr<IIntegrator> ig) {
    integrator = std::move(ig);
    // std::cout << "Integrator set." << std::endl; 
}

bool Simulation::is_cuda_backend() const {
#ifdef USE_CUDA
    return dynamic_cast<const CudaN2Backend*>(physics_backend.get()) != nullptr;
#else
    return false;
#endif
}

void Simulation::run_simulation(double total_simulation_time_s, int max_sim_steps) {
    if (!physics_backend || !integrator) {
        std::cerr << "Error: Physics backend or integrator not set." << std::endl;
        return;
    }
    if (particles.capacity == 0 || particles.current_num_particles == 0) {
        std::cerr << "Error: Particles not initialized or capacity is zero. Call particles.initialize_storage() and an InitialConditions function." << std::endl;
        return;
    }
    
    // If physics_backend was set *before* particles were initialized, initialize it now.
    if (physics_backend && particles.capacity > 0) {
        physics_backend->initialize(&particles);
    }


    // Initial GPU setup if using CUDA backend
    if (is_cuda_backend()) {
        std::cout << "CUDA backend detected. Allocating GPU memory and copying initial data..." << std::endl;
        particles.allocate_gpu_memory(); 
        particles.copy_all_to_gpu();     
    }

    std::cout << "Computing initial accelerations a(0)..." << std::endl;
    physics_backend->compute_accelerations(); 
    if (is_cuda_backend()) {
        particles.copy_acc_from_gpu(); 
    }

    update_adaptive_timestep();
    std::cout << "Initial adaptive dt: " << dt_adaptive << " s." << std::endl;

    current_time = 0.0;
    step_count = 0;
    double next_output_time = 0.0;
    if (config_dt_output > 0) { // Output initial state
        output_snapshot(step_count);
        next_output_time = config_dt_output;
    }


    auto simulation_start_chrono = std::chrono::high_resolution_clock::now();

    while (current_time < total_simulation_time_s && (max_sim_steps == -1 || step_count < max_sim_steps)) {
        run_single_step();

        if (config_dt_output > 0 && current_time >= next_output_time) {
            output_snapshot(step_count);
            next_output_time += config_dt_output; 
        }
        
        if (particles.num_active_particles <= 1 && particles.current_num_particles > 1) {
            std::cout << "Simulation ended early: only " << particles.num_active_particles << " active particle(s) remaining." << std::endl;
            break;
        }
    }

    auto simulation_end_chrono = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sim_duration = simulation_end_chrono - simulation_start_chrono;

    std::cout << "\n--- Simulation Finished ---" << std::endl;
    std::cout << "Total steps: " << step_count << std::endl;
    std::cout << "Final simulation time: " << current_time / (2 * PI_CONST) << " years"
              << " (" << current_time << " s)" << std::endl;
    std::cout << "Total wall-clock time: " << sim_duration.count() << " s." << std::endl;
    if (step_count > 0) {
        std::cout << "Average wall-clock time per step: " << sim_duration.count() / step_count << " s." << std::endl;
    }
    
    if (config_dt_output > 0) { // Ensure final snapshot is written if not perfectly aligned
        output_snapshot(step_count); 
    }
}


void Simulation::run_single_step() {
    auto step_start_chrono = std::chrono::high_resolution_clock::now();

    // CPU Integrator (KDK) uses a(t) from CPU. Modifies x(t), v(t) on CPU.
    // Then calls GPU Physics Backend.
    // GPU Physics Backend needs x(t+dt) (from CPU KDK drift) on GPU.
    // So, sync before integrator->step is NOT what KDK needs for its *first* part.
    // KDK step:
    // 1. CPU: v(t+dt/2) = v(t) + a(t)*dt/2  (uses CPU particles.acc[a(t)])
    // 2. CPU: x(t+dt) = x(t) + v(t+dt/2)*dt
    // 3. GPU Physics: needs x(t+dt).
    //    Inside CudaN2Backend::compute_accelerations():
    //        Copy CPU particles.posX (now x(t+dt)) to d_posX.
    //        Compute d_acc (now a(t+dt) on GPU).
    // 4. Simulation loop: Copy d_acc (a(t+dt) on GPU) to CPU particles.acc.
    // 5. CPU: v(t+dt) = v(t+dt/2) + particles.acc[a(t+dt)]*dt/2

    // Call integrator. It will handle calls to physics_backend.
    // The integrator expects CPU particles.acc to contain a(t).
    integrator->step(particles, dt_adaptive, physics_backend.get());
    // After this returns:
    // - CPU particles.posX/velX are updated to t+dt.
    // - If physics_backend was GPU, d_acc contains a(t+dt). CPU particles.acc is still a(t).

    if (is_cuda_backend()) {
        // Sync a(t+dt) from GPU (where physics_backend computed it) to CPU.
        particles.copy_acc_from_gpu();
        // Now CPU particles.acc also contains a(t+dt).
        // The KDK integrator's second kick (if it happened after the physics_backend call *within* integrator->step)
        // would have used this now-synced CPU a(t+dt).
        // This sequence means the KDK's second kick used the correct new accelerations.
    }
    // At this point, CPU: x(t+dt), v(t+dt), a(t+dt) are all consistent.
    // GPU: d_posX, d_velX still reflect pre-integrator state or state after first KDK parts if synced.
    //      d_acc has a(t+dt).

    // --- Collision Detection and Handling ---
    std::vector<CollisionPair> collision_pairs;
    if (is_cuda_backend()) {
        // Collision detection needs up-to-date positions x(t+dt) on GPU.
        particles.copy_pos_vel_mass_radius_active_id_to_gpu(); // This copies x(t+dt), v(t+dt) from CPU to GPU
    }
    // Now GPU d_posX has x(t+dt).
    collision_pairs = physics_backend->detect_collisions(); // Uses d_posX if GPU backend.
                                                            // Returns CPU vector.

    if (!collision_pairs.empty()) {
        bool merged_this_step = false;
        for (const auto& pair_idx_info : collision_pairs) {
            int p1_current_idx = pair_idx_info.p1_idx;
            int p2_current_idx = pair_idx_info.p2_idx;
            bool result = particles.merge(p1_current_idx, p2_current_idx);
            merged_this_step = merged_this_step || result;
        }
        if (merged_this_step) {
            // CPU state changed due to merge. GPU state is now stale.
            // Recompute accelerations a(t+dt) for the new merged configuration.
            if (is_cuda_backend()) {
                particles.copy_all_to_gpu(); // Sync all CPU changes (merged state) to GPU
            }
            // Now GPU d_posX, d_mass, etc. reflect merged state.
            physics_backend->compute_accelerations(); // Recomputes a(t+dt) on GPU, stores in d_acc
            if (is_cuda_backend()) {
                particles.copy_acc_from_gpu(); // Sync new a(t+dt) from GPU to CPU
            }
            // Now CPU and GPU a(t+dt) are consistent with merged state.
        }
    }
    
    current_time += dt_adaptive;
    step_count++;
    update_adaptive_timestep(); // Uses CPU a(t+dt) for estimation.

    auto step_end_chrono = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> step_duration_ms = step_end_chrono - step_start_chrono;

    if (step_count % 100 == 0) { 
        std::cout << "Step: " << std::setw(7) << step_count
                  << ", Time: " << std::fixed << std::setprecision(4) << current_time / (2 * PI_CONST) << " yr"
                  << ", dt: " << std::scientific << std::setprecision(3) << dt_adaptive << " s"
                  << ", Active: " << std::setw(6) << particles.num_active_particles
                  << ", Step Walltime: " << std::fixed << std::setprecision(2) << step_duration_ms.count() << " ms"
                  << std::defaultfloat << std::endl;
    }
    // char c = getchar(); // plays the role of system("pause")
}


void Simulation::update_adaptive_timestep() {
    if (particles.num_active_particles == 0) {
        dt_adaptive = config_dt_max; 
        return;
    }
    // This estimate is based on CPU data (particles.accX etc. should hold a(t+dt))
    // If estimate_min_dt_component itself is a GPU method, it needs d_acc to be synced.
    // Current CpuN2Backend::estimate_min_dt_component uses CPU data.
    // If CudaN2Backend::estimate_min_dt_component is used, it needs d_acc, d_pos, d_vel.
    // These should be up-to-date on GPU from previous syncs if collision handling happened.
    if (is_cuda_backend()) {
        // Ensure GPU has latest state for dt estimation if physics_backend::estimate_min_dt_component is GPU-based
         particles.copy_all_to_gpu(); // Safest, ensures pos, vel, acc are synced H->D
    }

    double estimated_dt_component = physics_backend->estimate_min_dt_component(config_adaptive_dt_safety_factor);
    
    // Simplified dt adjustment
    dt_adaptive = estimated_dt_component; // Directly use the estimate (with safety factor already applied)
                                          // Could add more complex logic for smoother changes.
    
    dt_adaptive = std::max(config_dt_min, std::min(config_dt_max, dt_adaptive));
}


void Simulation::output_snapshot(int step_num) {
    std::cout << "\n--- Snapshot at Step: " << std::setw(7) << step_num 
              << ", Sim Time: " << std::fixed << std::setprecision(4) 
              << current_time / (2 * PI_CONST) << " years ---" << std::defaultfloat << std::endl;
    std::cout << "Active Particles: " << particles.num_active_particles << std::endl;
    std::cout << "Current dt: " << std::scientific << std::setprecision(3) << dt_adaptive << " s" 
              << std::defaultfloat << std::endl;

    int print_count = 0;
    const int max_to_print = std::min(10, static_cast<int>(particles.num_active_particles));
    if (particles.num_active_particles > 0) {
         std::cout << "  First " << max_to_print << " active particles (ID, mass_kg, radius_m, pos_AU(x,y,z), vel_m/s(x,y,z)):" << std::endl;
    }

    for (size_t i = 0; i < particles.capacity && print_count < max_to_print; ++i) {
        if (particles.active[i]) {
            std::cout << "  P" << std::setw(4) << particles.id[i] << " (idx " << std::setw(4) << i << "): "
                      << "m=" << std::scientific << std::setprecision(2) << particles.mass[i] << std::defaultfloat
                      << " r=" << std::fixed << std::setprecision(0) << particles.radius[i]
                      << " pos=(" << std::fixed << std::setprecision(3) << particles.posX[i] / AU << ", "
                                  << particles.posY[i] / AU << ", "
                                  << particles.posZ[i] / AU << ")"
                      << " vel=(" << std::scientific << std::setprecision(2) << particles.velX[i] << ", "
                                  << particles.velY[i] << ", "
                                  << particles.velZ[i] << ")"
                      << std::defaultfloat << std::endl;
            print_count++;
        }
    }
    if (particles.num_active_particles > max_to_print) {
        std::cout << "  ... and " << (particles.num_active_particles - max_to_print) << " more." << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------\n" << std::endl;
}