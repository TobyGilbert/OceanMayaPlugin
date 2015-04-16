#ifndef OCEANCUDA_H
#define OCEANCUDA_H
//----------------------------------------------------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <helper_cuda.h>
//----------------------------------------------------------------------------------------------------------------------
/// @brief Takes in array of frequency domain at time zero and output a frequency domain at any time
/// @param d_H0 an array of points in the frequency domain at time zero
/// @param d_Ht array of points in the frequnecy domain used for generating heights
/// @param _time the current time of the simulation
/// @param _res the resolution of the ocean grid
//----------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2 *d_H0, float2 *d_Ht, float _time, int _res);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Uses the frequency domain of the heightfield to generate horizontal displacement in the frequency domain
/// @param d_Ht array of points in the frequnecy domain used for generating heights
/// @param d_ChopX an array to output displacement in the x axis
/// @param d_ChopZ an array to output displacement in the z axis
//----------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Ht, float2 *d_ChopX, float2 *d_ChopZ, int _res, float2 _windSpeed);
//----------------------------------------------------------------------------------------------------------------------
#endif
//----------------------------------------------------------------------------------------------------------------------
