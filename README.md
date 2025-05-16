# Asteroid-Formation-Refactor

nbody-asteroids/
├── CMakeLists.txt
├── README.md
├── config/
│   └── simulation_params.json  // Example config file
├── data/
│   ├── initial_conditions/
│   └── output/                 // Simulation snapshots
├── src/
│   ├── main.cpp
│   ├── particle_data.h
│   ├── particle_data.cpp
│   ├── dsu.h
│   ├── dsu.cpp                 // (Simple enough might be header-only)
│   ├── constants.h             // Physical constants (G, etc.)
│   ├── io_utils.h              // For reading configs, writing output
│   ├── io_utils.cpp
│   ├── cuda_utils.h            // CUDA error checking macros, device query
│   ├── cuda_utils.cu
│   ├── backends/
│   │   ├── iphysics_backend.h  // Interface for physics computations
│   │   ├── cpu_n2_backend.h
│   │   ├── cpu_n2_backend.cpp
│   │   ├── cuda_n2_backend.h
│   │   ├── cuda_n2_backend.cu  // CUDA kernels here
│   │   └── cpu_spatial_hash_backend.h
│   │   └── cpu_spatial_hash_backend.cpp
│   ├── integrator.h            // E.g., LeapfrogIntegrator
│   ├── integrator.cpp
│   ├── collision_handler.h
│   ├── collision_handler.cpp   // Merging logic
│   ├── simulation.h
│   └── simulation.cpp          // Main simulation loop orchestration
└── third_party/                // Optional: for JSON parsers, etc.