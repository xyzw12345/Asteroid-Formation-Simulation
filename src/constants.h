#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath> // For M_PI if needed, or other math consts

// Physical Constants
constexpr double G_CONST = 1; // Gravitational constant (m^3 kg^-1 s^-2)
constexpr double AU = 1;        // Astronomical Unit (meters)
constexpr double PI_CONST = 3.141592653589793238;

// Simulation Parameters
// Softening length squared to prevent singularities in N-body gravity.
// Value depends on typical scales of your simulation.
// For example, if typical radii are ~1km, eps^2 might be (100m)^2 = 1e4 m^2.
// Or a fraction of the smallest particle radius squared.
// This needs to be chosen carefully. Too large, and it alters dynamics. Too small, and extreme forces still occur.
constexpr double SOFTENING_EPSILON_SQUARED = 1.0e-12; // (Example: (1km)^2, if dealing with km-scale objects)
                                                 // Adjust this based on your particle sizes and distances!

#endif // CONSTANTS_H