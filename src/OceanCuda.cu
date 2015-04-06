#include "OceanCuda.h"

__global__ void fillKernel(float* array) {
    array[threadIdx.x] = threadIdx.x * 0.5;
}

void fillGpuArray(float* array, int count) {
    fillKernel<<<1, count>>>(array);
}

// ----------------------------------------------------------------------------------------------------------------------------------------
/// @author Toby Gilbert
// ----------------------------------------------------------------------------------------------------------------------------------------
#include <helper_cuda.h>
#include <cufft.h>
#include "OceanCuda.h"
#include <glm/glm.hpp>
#include <thrust/complex.h>
#include <curand.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Given a time you can create a field of frequency amplitudes
/// @param d_h0Pointer An OpenGL buffer which stores a set of amplitudes and phases at time zero
/// @param d_htPointer An OpenGL buffer for outputting the frequency amplitude field
/// @param _time The current simulation time
/// @param _res The simulation resolution
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void frequencyDomain(float2* d_h0Pointer, float2* d_htPointer, float _time, int _res){
    // A constant for the accelleration due to gravity
    const float g = 9.81;

    // A 2D vector to represent a position on the grid with constraits -(_res/2) <= k < (_res/2)
    glm::vec2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0));
    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0));
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
    hStar.y = (h0conjugate.x * complexExpConjugate.x - h0conjugate.y * complexExpConjugate.y) ;

    // Output h(k,t) term to d_htPointer buffer which represents a set of points in the frequency domain
    float2 hTilde;
    hTilde.x = h.x + hStar.x;
    hTilde.y = h.y + hStar.y;
    d_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].x = hTilde.x;
    d_htPointer[(blockIdx.x * blockDim.x) + threadIdx.x].y = hTilde.y;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Once inverse FFT has been performed points in the frequency domain are converted to the spatial domain
/// and can be used to update the heights
/// @param d_position An OpenGL buffer for storing the current positions of the vertices in the grid
/// @param d_height An OpenGL buffer which holds the new heights of grid positions
/// @param d_normal An OpenGL buffer which holds the normals
/// @param d_xDisplacement An OpenGL buffer for storing the displacment in the x axis
/// @param _res The resolution of the grid
/// @param _scale Scales the amplitude of the waves
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void height(glm::vec3* d_position, cudaSurfaceObject_t _surface, float2* d_height, float2* d_xDisplacement, float _choppiness, int _res, float _scale){
    // A vertex on the grid
    int u = int(threadIdx.x - (_res * floor(double(threadIdx.x / _res))));
    int v = int((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res)));


    // Sign correction - Unsure why this is needed
    float sign = 1.0;
    if ((u+v) % 2 != 0){
        sign = -1.0;
    }
    // Update the heights of the vertices and add x and z displacement
    d_position[(blockIdx.x * blockDim.x) + threadIdx.x].y = (d_height[(blockIdx.x * blockDim.x) + threadIdx.x].x / _scale) * sign ;
    // Write the heights to a texture so it can be used in the shader
//    surf2Dwrite(make_uchar4((d_height[(blockIdx.x * blockDim.x) + threadIdx.x].x / _scale) * sign + 100.0, 0, 0, 255), _surface, (int)sizeof(uchar4)*u, v, cudaBoundaryModeZero);
    d_position[(blockIdx.x * blockDim.x) + threadIdx.x].x += (d_xDisplacement[(blockIdx.x * blockDim.x) + threadIdx.x].x / _scale) * _choppiness * sign;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
/// @brief Create x displacement in in the frequency domain
/// @param
/// @param d_xDisplacement An OpenGL buffer to store the x displacement in the frequency domain
/// @param d_zDisplacement An OpenGL buffer to store the z displacement in the frequency domain
/// @param _res The resolution of the grid
// ----------------------------------------------------------------------------------------------------------------------------------------
__global__ void choppiness(float2* d_ht, int _res){
    // k - A position on the grid
    float2 k;
    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0));
    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0));
    float kLen = sqrt(double(k.x*k.x + k.y*k.y));
    float2 prev = d_ht[(blockIdx.x * blockDim.x) + threadIdx.x];

    float Kx = k.x / kLen;
    float Kz = k.y / kLen;
    if (kLen == 0.0){
        Kx = 0.0;
        Kz = 0.0;
    }
    d_ht[(blockIdx.x * blockDim.x) + threadIdx.x].x = prev.x * -Kx;
}

__global__ void choppinessX(float2* d_Heights, float2* d_ChopX, float2* d_ChopZ, int _res, float2 _windSpeed){
    // A vertex on the grid
    int u = int(threadIdx.x - (_res * floor(double(threadIdx.x / _res))));
    int v = int((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res)));

    float2 k;
//    k.x = float((threadIdx.x - (_res * floor(double(threadIdx.x / _res)))) - (_res/2.0));
//    k.y = float(((blockIdx.x * (blockDim.x/(float)_res)) + ceil(double(threadIdx.x / _res))) - (_res/2.0));
    k.x = _windSpeed.x;
    k.y = _windSpeed.y;

    float kLen = sqrt(double(k.x*k.x + k.y*k.y));

    float Kx = k.x / kLen;
    float Kz = k.y / kLen;

    if (kLen == 0.0){
        Kx = 0.0;
        Kz = 0.0;
    }

    d_ChopX[(blockIdx.x * blockDim.x) + threadIdx.x].x = (d_Heights[(blockIdx.x * blockDim.x) + threadIdx.x].x / 50000.0) * -Kx;
    d_ChopZ[(blockIdx.x * blockDim.x) + threadIdx.x].x = (d_Heights[(blockIdx.x * blockDim.x) + threadIdx.x].x / 50000.0) * -Kz;

}

// ----------------------------------------------------------------------------------------------------------------------------------------
void updateFrequencyDomain(float2* d_h0, float2* d_ht, float _time, int _res){
    int numBlocks =( _res * _res )/ 1024;
    frequencyDomain<<<numBlocks, 1024>>>(d_h0, d_ht, _time, _res);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void updateHeight(glm::vec3* d_position, cudaSurfaceObject_t _surface, float2* d_height, float2 *d_xDisplacement, float _choppiness, int _res, float _scale){

    // Bind the cudaArray to a globally scoped CUDA surface
//    cudaBindSurfaceToArray(_surface, d_heightsCudaArray);

    int numBlocks =( _res * _res )/ 1024;
    height<<<numBlocks, 1024>>>(d_position, _surface, d_height, d_xDisplacement, _choppiness,  _res, _scale);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
void addChoppiness(float2* d_Heights, float2* d_ChopX, float2* d_ChopZ, int _res, float2 _windSpeed){
    int numBlocks =( _res * _res )/ 1024;
    choppinessX<<<numBlocks, 1024>>>(d_Heights, d_ChopX, d_ChopZ, _res, _windSpeed);
}
// ----------------------------------------------------------------------------------------------------------------------------------------
