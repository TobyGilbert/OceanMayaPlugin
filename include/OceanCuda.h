#ifndef OCEANCUDA_H
#define OCEANCUDA_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

void fillGpuArray(float* array, int count);

void updateFrequencyDomain(float2 *d_h0, float2 *d_ht, float _time, int _res);

void updateHeight(float3* d_position, cudaSurfaceObject_t _surface, float2* d_height, float2 *d_xDisplacement, float _choppiness, int _res, float _scale);

void addChoppiness(float2* d_Heights, float2 *d_ChopX, float2 *d_ChopZ, int _res, float2 _windSpeed);


#endif
