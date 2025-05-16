#ifndef PARTICLE_DATA_H
#define PARTICLE_DATA_H

#include <vector>
#include <string>
#include <numeric> // For std::iota

// Forward declaration for CUDA error checking utilities
#ifdef USE_CUDA
// No need to include full cuda_runtime.h here if cuda_utils.h handles it
#endif

class ParticleData {
public:
    // Basic properties (SoA) - all sized to 'capacity'
    std::vector<double> posX, posY, posZ;
    std::vector<double> velX, velY, velZ;
    std::vector<double> accX, accY, accZ;
    std::vector<double> mass;
    std::vector<double> radius;

    // Simulation state - all sized to 'capacity'
    std::vector<int> id;          // Unique, persistent ORIGINAL ID (0 to capacity-1 usually)
    std::vector<bool> active;     // True if particle at this array index is currently active

    int num_active_particles = 0; // Count of true values in 'active' vector

    // DSU (Disjoint Set Union) data - operates on ORIGINAL IDs
    // Sized to 'capacity' (max number of original particles)
    std::vector<int> dsu_parent;
    std::vector<int> dsu_set_size; // Stores size of the set if this ID is a representative

    // GPU mirror (pointers to device memory) - allocated for 'capacity' elements
    double *d_posX = nullptr, *d_posY = nullptr, *d_posZ = nullptr;
    double *d_velX = nullptr, *d_velY = nullptr, *d_velZ = nullptr;
    double *d_accX = nullptr, *d_accY = nullptr, *d_accZ = nullptr;
    double *d_mass = nullptr;
    double *d_radius = nullptr;
    int    *d_id = nullptr;
    bool   *d_active = nullptr;

    size_t capacity = 0;        // Max number of particles storage allocated for (and DSU size)
    size_t current_num_particles = 0; // Number of particles actually initialized via add_particle.
                                      // This is how many slots from 0 to current_num_particles-1 hold valid initial data.
                                      // Iterations on CPU often go up to this value if not checking active flags for all 'capacity'.

    ParticleData() = default;
    ~ParticleData(); // To handle GPU memory cleanup

    // Allocates memory for all CPU vectors and initializes DSU.
    // Does NOT allocate GPU memory (use allocate_gpu_memory for that).
    void initialize_storage(size_t initial_capacity);
    
    // Adds a new particle's properties to the next available slot (index = current_num_particles).
    // Assigns a unique original_id (particles.id[index]).
    // Returns the array index of the newly added particle.
    int add_particle(double pX, double pY, double pZ,
                     double vX, double vY, double vZ,
                     double m, double r);

    void clear_accelerations_cpu();    // Zeroes out accX, accY, accZ on CPU for 'current_num_particles'
    void clear_accelerations_gpu();    // Zeroes out d_accX, d_accY, d_accZ on GPU for 'capacity'

    // DSU operations (operating on original_ids)
    // Finds the representative of the set containing 'original_particle_id'. Performs path compression.
    int find_set_representative(int original_particle_id);
    // Unites the sets containing 'original_particle_id1' and 'original_particle_id2'. Uses union by size.
    // Returns true if a union occurred, false if they were already in the same set.
    bool merge(int original_particle_id1, int original_particle_id2);

    // Utility: Recalculates 'num_active_particles' by iterating through the 'active' vector up to 'capacity'.
    void count_active_particles();
    
    // GPU data management
    void allocate_gpu_memory(); // Allocates GPU memory based on current 'capacity'
    void free_gpu_memory();

    // Copy data between Host (CPU) and Device (GPU)
    // These typically copy 'capacity' elements because GPU kernels often iterate over 'capacity'.
    void copy_all_to_gpu();
    void copy_pos_vel_mass_radius_active_id_to_gpu();
    void copy_acc_to_gpu(); // Host to Device
    
    void copy_all_from_gpu();
    void copy_pos_vel_mass_radius_active_id_from_gpu();
    void copy_acc_from_gpu(); // Device to Host

private:
    int next_available_original_id = 0; // Counter to assign unique original IDs (0 upwards)
};

#endif // PARTICLE_DATA_H