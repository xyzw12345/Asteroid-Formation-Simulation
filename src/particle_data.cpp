#include "particle_data.h"
#include "cuda_utils.h" // For CUDA_CHECK and conditional CUDA compilation definition
#include <stdexcept>    // For std::runtime_error
#include <algorithm>    // For std::fill
#include <iostream>     // For std::cerr

#ifdef USE_CUDA
#include <cuda_runtime.h> // For cudaMemset, cudaMalloc, cudaFree, cudaMemcpy
#endif

ParticleData::~ParticleData() {
    free_gpu_memory();
}

void ParticleData::initialize_storage(size_t initial_capacity) {
    if (capacity > 0 && initial_capacity != capacity) {
        // If re-initializing with a different capacity, ensure GPU memory is freed
        // as it's tied to the old capacity.
        free_gpu_memory();
    }

    capacity = initial_capacity;
    current_num_particles = 0;
    num_active_particles = 0;
    next_available_original_id = 0;

    if (capacity == 0) { // Handle case of zero capacity
        posX.clear(); posY.clear(); posZ.clear();
        velX.clear(); velY.clear(); velZ.clear();
        accX.clear(); accY.clear(); accZ.clear();
        mass.clear(); radius.clear();
        id.clear(); active.clear();
        dsu_parent.clear(); dsu_set_size.clear();
        return;
    }
    
    // Resize all data vectors to full capacity
    // Initialize with default values (0.0 for double, false for bool, -1 for id)
    posX.assign(capacity, 0.0); posY.assign(capacity, 0.0); posZ.assign(capacity, 0.0);
    velX.assign(capacity, 0.0); velY.assign(capacity, 0.0); velZ.assign(capacity, 0.0);
    accX.assign(capacity, 0.0); accY.assign(capacity, 0.0); accZ.assign(capacity, 0.0);
    mass.assign(capacity, 0.0); radius.assign(capacity, 0.0);
    id.assign(capacity, -1);      // Initialize original IDs to an invalid state
    active.assign(capacity, false); // All slots initially inactive by default

    // Initialize DSU arrays for the full capacity
    dsu_parent.resize(capacity);
    std::iota(dsu_parent.begin(), dsu_parent.end(), 0); // Each element is its own parent
    dsu_set_size.assign(capacity, 1);                   // Each set initially has size 1
}

int ParticleData::add_particle(double pX, double pY, double pZ,
                               double vX, double vY, double vZ,
                               double m, double r) {
    if (current_num_particles >= capacity) {
        throw std::runtime_error("ParticleData: Exceeded pre-allocated capacity. Cannot add more particles.");
    }
    if (next_available_original_id >= static_cast<int>(capacity)) {
        // This case implies we are trying to assign an original_id that's too large for DSU arrays
        throw std::runtime_error("ParticleData: Ran out of unique original IDs for the given DSU capacity.");
    }

    int particle_idx = current_num_particles; // The new particle will occupy this array index

    posX[particle_idx] = pX; posY[particle_idx] = pY; posZ[particle_idx] = pZ;
    velX[particle_idx] = vX; velY[particle_idx] = vY; velZ[particle_idx] = vZ;
    // accX, accY, accZ are already 0.0 from assign in initialize_storage for this slot
    mass[particle_idx] = m;
    radius[particle_idx] = r;
    
    active[particle_idx] = true; // Mark this slot as active
    id[particle_idx] = next_available_original_id; // Assign the persistent original ID

    // DSU structure for 'next_available_original_id' was already set up in initialize_storage
    
    next_available_original_id++;
    current_num_particles++; // Increment count of initialized particle slots
    num_active_particles++;  // Increment count of active particles

    return particle_idx; // Return the array index where this particle was added
}

void ParticleData::clear_accelerations_cpu() {
    // Only clear for particles that have been added, up to current_num_particles
    // Or clear full capacity if accelerations might be non-zero beyond current_num_particles
    // For safety and consistency with GPU (which clears full capacity), let's clear up to capacity.
    std::fill(accX.begin(), accX.end(), 0.0); // .end() refers to capacity
    std::fill(accY.begin(), accY.end(), 0.0);
    std::fill(accZ.begin(), accZ.end(), 0.0);
}

void ParticleData::clear_accelerations_gpu() {
#ifdef USE_CUDA
    if (d_accX && capacity > 0) {
        CUDA_CHECK(cudaMemset(d_accX, 0, capacity * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_accY, 0, capacity * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_accZ, 0, capacity * sizeof(double)));
    }
#else
    // No-op or print warning if called without CUDA enabled
#endif
}

int ParticleData::find_set_representative(int original_particle_id) {
    if (original_particle_id < 0 || original_particle_id >= static_cast<int>(dsu_parent.size())) {
        std::cerr << "Error: find_set_representative called with out-of-bounds original_id: " 
                  << original_particle_id << " (DSU size: " << dsu_parent.size() << ")" << std::endl;
        // For robustness, one might throw an exception or return a specific error code.
        // This often indicates a logic error elsewhere if a bad ID is passed.
        if (dsu_parent.empty()) return -1; // Avoid crashing if DSU is empty
        return (original_particle_id < 0) ? 0 : static_cast<int>(dsu_parent.size()) - 1; // Clamp to prevent crash, but it's an error
    }
    if (dsu_parent[original_particle_id] == original_particle_id) {
        return original_particle_id;
    }
    return dsu_parent[original_particle_id] = find_set_representative(dsu_parent[original_particle_id]); // Path compression
}

bool ParticleData::merge(int original_particle_id1, int original_particle_id2) {
    if (original_particle_id1 < 0 || original_particle_id1 >= static_cast<int>(dsu_parent.size()) ||
        original_particle_id2 < 0 || original_particle_id2 >= static_cast<int>(dsu_parent.size())) {
        std::cerr << "Error: unite_sets called with out-of-bounds original_id." << std::endl;
        return false;
    }

    int root1 = find_set_representative(original_particle_id1);
    int root2 = find_set_representative(original_particle_id2);

    if (root1 != root2) {
        // Union by size: attach smaller tree under root of larger tree
        if (dsu_set_size[root1] < dsu_set_size[root2]) {
            std::swap(root1, root2);
        }

        dsu_parent[root2] = root1;
        dsu_set_size[root1] += dsu_set_size[root2];

        double m_survivor = particles.mass[root1];
        double m_victim = particles.mass[root2];
        double m_new_total = m_survivor + m_victim;

        // 1. Conserve Momentum: new_vel = (m1*v1 + m2*v2) / (m1+m2)
        //    New velocity for the survivor.
        particles.velX[root1] = (m_survivor * particles.velX[root1] + m_victim * particles.velX[root2]) / m_new_total;
        particles.velY[root1] = (m_survivor * particles.velY[root1] + m_victim * particles.velY[root2]) / m_new_total;
        particles.velZ[root1] = (m_survivor * particles.velZ[root1] + m_victim * particles.velZ[root2]) / m_new_total;

        // 2. New Position (Center of Mass): new_pos = (m1*p1 + m2*p2) / (m1+m2)
        //    New position for the survivor.
        particles.posX[root1] = (m_survivor * particles.posX[root1] + m_victim * particles.posX[root2]) / m_new_total;
        particles.posY[root1] = (m_survivor * particles.posY[root1] + m_victim * particles.posY[root2]) / m_new_total;
        particles.posZ[root1] = (m_survivor * particles.posZ[root1] + m_victim * particles.posZ[root2]) / m_new_total;
        
        // 3. New Mass:
        particles.mass[root1] = m_new_total;

        // 4. New Radius:
        //    Assuming constant density and spherical particles: Volume_new = Volume1 + Volume2
        //    (4/3)*pi*r_new^3 = (4/3)*pi*r1^3 + (4/3)*pi*r2^3
        //    r_new^3 = r1^3 + r2^3  => r_new = cbrt(r1^3 + r2^3)
        double r1_cubed = particles.radius[root1] * particles.radius[root1] * particles.radius[root1];
        double r2_cubed = particles.radius[root2] * particles.radius[root2] * particles.radius[root2];
        particles.radius[root1] = std::cbrt(r1_cubed + r2_cubed);

        // 5. Mark Victim as Inactive:
        particles.active[root2] = false;

        particles.num_active_particles--;

        return true; // Union occurred
    }
    return false; // Already in the same set
}

void ParticleData::count_active_particles() {
    num_active_particles = 0;
    // Iterate up to full capacity as 'active' array is sized to capacity
    for (size_t i = 0; i < capacity; ++i) {
        if (active[i]) {
            num_active_particles++;
        }
    }
}

// --- GPU Data Management ---
void ParticleData::allocate_gpu_memory() {
#ifdef USE_CUDA
    if (capacity == 0) return;

    free_gpu_memory(); // Ensure any old allocations are freed

    size_t N_bytes_double = capacity * sizeof(double);
    size_t N_bytes_int    = capacity * sizeof(int);
    size_t N_bytes_bool   = capacity * sizeof(bool);

    CUDA_CHECK(cudaMalloc(&d_posX, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_posY, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_posZ, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_velX, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_velY, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_velZ, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_accX, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_accY, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_accZ, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_mass, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_radius, N_bytes_double));
    CUDA_CHECK(cudaMalloc(&d_id, N_bytes_int));
    CUDA_CHECK(cudaMalloc(&d_active, N_bytes_bool));
#else
    if (capacity > 0) {
        // std::cerr << "Warning: ParticleData::allocate_gpu_memory() called but USE_CUDA is not defined." << std::endl;
    }
#endif
}

void ParticleData::free_gpu_memory() {
#ifdef USE_CUDA
    if (d_posX)   { cudaFree(d_posX); d_posX = nullptr; } // Check before free is good practice
    if (d_posY)   { cudaFree(d_posY); d_posY = nullptr; }
    if (d_posZ)   { cudaFree(d_posZ); d_posZ = nullptr; }
    if (d_velX)   { cudaFree(d_velX); d_velX = nullptr; }
    if (d_velY)   { cudaFree(d_velY); d_velY = nullptr; }
    if (d_velZ)   { cudaFree(d_velZ); d_velZ = nullptr; }
    if (d_accX)   { cudaFree(d_accX); d_accX = nullptr; }
    if (d_accY)   { cudaFree(d_accY); d_accY = nullptr; }
    if (d_accZ)   { cudaFree(d_accZ); d_accZ = nullptr; }
    if (d_mass)   { cudaFree(d_mass); d_mass = nullptr; }
    if (d_radius) { cudaFree(d_radius); d_radius = nullptr; }
    if (d_id)     { cudaFree(d_id); d_id = nullptr; }
    if (d_active) { cudaFree(d_active); d_active = nullptr; }
#endif
}

void ParticleData::copy_all_to_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_posX) return; // No data or GPU memory not allocated

    size_t N_bytes_double = capacity * sizeof(double);
    size_t N_bytes_int    = capacity * sizeof(int);
    size_t N_bytes_bool   = capacity * sizeof(bool);

    CUDA_CHECK(cudaMemcpy(d_posX, posX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, posY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, posZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, velX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, velY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, velZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accX, accX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accY, accY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accZ, accZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, mass.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radius, radius.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_id, id.data(), N_bytes_int, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, active.data(), N_bytes_bool, cudaMemcpyHostToDevice));
#endif
}

void ParticleData::copy_pos_vel_mass_radius_active_id_to_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_posX) return;

    size_t N_bytes_double = capacity * sizeof(double);
    size_t N_bytes_int    = capacity * sizeof(int);
    size_t N_bytes_bool   = capacity * sizeof(bool);

    CUDA_CHECK(cudaMemcpy(d_posX, posX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, posY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, posZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, velX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, velY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, velZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, mass.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radius, radius.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_id, id.data(), N_bytes_int, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, active.data(), N_bytes_bool, cudaMemcpyHostToDevice));
#endif
}

void ParticleData::copy_acc_to_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_accX) return;
    size_t N_bytes_double = capacity * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_accX, accX.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accY, accY.data(), N_bytes_double, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_accZ, accZ.data(), N_bytes_double, cudaMemcpyHostToDevice));
#endif
}


void ParticleData::copy_all_from_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_posX) return; // No data or GPU memory not allocated

    size_t N_bytes_double = capacity * sizeof(double);
    size_t N_bytes_int    = capacity * sizeof(int);
    size_t N_bytes_bool   = capacity * sizeof(bool);

    CUDA_CHECK(cudaMemcpy(posX.data(), d_posX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), d_posY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posZ.data(), d_posZ, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velX.data(), d_velX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), d_velY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velZ.data(), d_velZ, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accX.data(), d_accX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accY.data(), d_accY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accZ.data(), d_accZ, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass.data(), d_mass, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(radius.data(), d_radius, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(id.data(), d_id, N_bytes_int, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(active.data(), d_active, N_bytes_bool, cudaMemcpyDeviceToHost));
#endif
}

void ParticleData::copy_pos_vel_mass_radius_active_id_from_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_posX) return;

    size_t N_bytes_double = capacity * sizeof(double);
    size_t N_bytes_int    = capacity * sizeof(int);
    size_t N_bytes_bool   = capacity * sizeof(bool);

    CUDA_CHECK(cudaMemcpy(posX.data(), d_posX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), d_posY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posZ.data(), d_posZ, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velX.data(), d_velX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velY.data(), d_velY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velZ.data(), d_velZ, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass.data(), d_mass, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(radius.data(), d_radius, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(id.data(), d_id, N_bytes_int, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(active.data(), d_active, N_bytes_bool, cudaMemcpyDeviceToHost));
#endif
}


void ParticleData::copy_acc_from_gpu() {
#ifdef USE_CUDA
    if (capacity == 0 || !d_accX) return;
    size_t N_bytes_double = capacity * sizeof(double);
    CUDA_CHECK(cudaMemcpy(accX.data(), d_accX, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accY.data(), d_accY, N_bytes_double, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accZ.data(), d_accZ, N_bytes_double, cudaMemcpyDeviceToHost));
#endif
}