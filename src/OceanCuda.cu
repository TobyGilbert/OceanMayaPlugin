// ----------------------------------------------------------------------------------------------------------------------------------------
#include "OceanCuda.h"
#include <helper_cuda.h>
#include <cufft.h>
#include <glm/glm.hpp>
#include <curand.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Given a time you can create a field of frequency amplitudes
/// @param d_h0Pointer an array of points in the frequency domain at time zero
/// @param d_htPointer an array used to output the points in the frequency domain at time _time
/// @param _time the current simulation time
/// @param _res the simulation resolution
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void frequencyDomain(float2* d_h0Pointer, float2* d_htPointer, float _time, int _res){
    // A constant for the accelleration due to gravity
    const float g = 9.81;

    // A 2D vector to represent a position on the grid with constraits -(_res/2) <= k < (_res/2)
    float2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0f));
    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0f));
    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    // Calculate the wave frequency
    float w = sqrt(double(g * kLen));

    // complexExp holds the complex exponential where the x value stores the real part and the y value stores the imaginary part
    float2 complexExp;
    complexExp.x = sin(w * _time);
    complexExp.y = cos(w * _time);

    float2 complexExpConjugate;
    complexExpConjugate.x = complexExp.x;
    complexExpConjugate.y = -complexExp.y;

    int blockNum =(( _res * _res )/ blockDim.x) - 1;

    float2 h0 = d_h0Pointer[(blockIdx.x * blockDim.x) + threadIdx.x];
    float2 h0conjugate = d_h0Pointer[((blockNum - blockIdx.x) * blockDim.x) + ((blockDim.x - 1) - threadIdx.x)];

    // Swap the imaginary parts sign
    h0conjugate.y = -h0conjugate.y;

    // Equation 26 of Tessendorf's paper h(k,t) = h0(k)exp{iw(k)t} + ~h0(-k)exp{-iw(k)t}
    float2 h;
    h.x = (h0.x * complexExp.x - h0.y * complexExp.y);
    h.y = (h0.x * complexExp.x + h0.y * complexExp.y);

    float2 hStar;
    hStar.x = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;
    hStar.y = (h0conjugate.x * complexExpConjugate.x + h0conjugate.y * complexExpConjugate.y) ;

    // Output h(k,t) term to d_htPointer buffer which represents a set of points in the frequency domain
    float2 hTilde;
    hTilde.x = h.x + hStar.x;
    hTilde.y = h.y + hStar.y;

    d_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x] = hTilde;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Create x displacement in in the frequency domain
/// @param d_Ht  array of points in the frequnecy domain used for generating heights
/// @param d_ChopX used for outputing displacement in the x axis in the frequency domain
/// @param d_ChopZ used for outputing displacement in the z axis in the frequency domain
/// @param _res the resolution of the ocean grid
/// @param _windSpeed the velocity of the wind
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void choppiness(float2* d_Ht, float2* d_ChopX, float2* d_ChopZ, int _res, float2 _windSpeed){
    // A vertex on the grid
    int u = int(threadIdx.x - (_res * floor(double(threadIdx.x / _res))));
    int v = int((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res)));

    float2 k;
    k.x = _windSpeed.x;
    k.y = _windSpeed.y;

    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    float Kx = k.x / kLen;
    float Kz = k.y / kLen;

    if (kLen == 0.0){
        Kx = 0.0;
        Kz = 0.0;
    }

    d_ChopX[(blockIdx.x * blockDim.x) + threadIdx.x].x = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].x * 0.0;
    d_ChopX[(blockIdx.x * blockDim.x) + threadIdx.x].y = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].y * -Kx;

    d_ChopZ[(blockIdx.x * blockDim.x) + threadIdx.x].x = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].x * 0.0;
    d_ChopZ[(blockIdx.x * blockDim.x) + threadIdx.x].y = d_Ht[(blockIdx.x * blockDim.x) + threadIdx.x].y * -Kz;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2* d_H0,  float2* d_Ht, float _time, int _res){
    int numBlocks =( _res * _res )/ 1024;
    frequencyDomain<<<numBlocks, 1024>>>(d_H0, d_Ht, _time, _res);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Ht, float2* d_ChopX, float2* d_ChopZ, int _res, float2 _windSpeed){
    int numBlocks =( _res * _res )/ 1024;
    choppiness<<<numBlocks, 1024>>>(d_Ht, d_ChopX, d_ChopZ, _res, _windSpeed);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
