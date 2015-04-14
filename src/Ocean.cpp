#include "Ocean.h"
#include "OceanCuda.h"
#include "mathsUtils.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <boost/random.hpp>
#include <helper_cuda.h>
//----------------------------------------------------------------------------------------------------------------------
Ocean::Ocean(){
    m_resolution = 128;
    m_gridSize = m_resolution * m_resolution;
    m_windDirection = make_float2(0.5, 0.5);
    m_A = 0.01;
    m_windSpeed = 100.0;
    m_L = (m_windSpeed* m_windSpeed) / 9.81;
    m_l = 1.0 / (float)m_L;
    initialise();
}
//----------------------------------------------------------------------------------------------------------------------
Ocean::~Ocean(){
    free(h_H0);
    free(h_Ht);
    free(h_Heights);
    checkCudaErrors(cudaFree(d_H0));
    checkCudaErrors(cudaFree(d_Ht));
    checkCudaErrors(cudaFree(d_Heights));
}
//----------------------------------------------------------------------------------------------------------------------
float Ocean::phillips(float2 _k){
    float kLen = sqrt(_k.x*_k.x + _k.y*_k.y);
    if (kLen == 0.0f){
        return 0.0f;
    }
    float ph = ( exp( -1 / ( (kLen * m_L )*(kLen * m_L ) ))  / pow(kLen, 4) );
    ph *= m_A;

    // | k . w |^2
    float kw = (normalise(_k).x * normalise(m_windDirection).x + normalise(_k).y * normalise(m_windDirection).y);
    ph *= kw * kw;

    if (kw < 0.0){
        ph *= 0.05;
    }
    // exp(-k^2 l^2)
    ph *= exp(-(kLen * kLen) * m_l * m_l);

    return ph;
}
// ----------------------------------------------------------------------------------------------------------------------------------------
// gaussian random number generator sourced from - NVIDIA OceanFFT
// Generates Gaussian random number with mean 0 and standard deviation 1.
// ----------------------------------------------------------------------------------------------------------------------------------------
float Ocean::gauss(){
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return sqrtf(-2 * logf(u1)) * cosf(2*M_PI * u2);
}
//----------------------------------------------------------------------------------------------------------------------
void Ocean::createH0(){
    // Assign memory on the host side and device to store h0 and evaluate
    for (int x=0; x<m_resolution; x++){
        for (int y=0; y<m_resolution; y++){
            int m = x - (m_resolution/2.0);
            int n = y - (m_resolution/2.0);
            float2 k;
            k.x = (M_2_PI * m) / 2000.0; //m_L;
            k.y = (M_2_PI * n) / 2000.0; //m_L;
            float2 h;

            h.x = (1.0/sqrt(2.0)) * gauss();
            if (k.x == 0 && k.y == 0){
                h.x = 0.0;
            }
            else{
                h.x *= sqrt(phillips(k));
            }

            h.y = (1.0/sqrt(2.0)) * gauss();
            if (k.x ==0 && k.y == 0){
                h.y = 0.0;
            }
            else{
                h.y *= sqrt(phillips(k));
            }
            h_H0[(y + (x * m_resolution))] = h;
        }
    }
    checkCudaErrors(cudaMemcpy(d_H0, h_H0, m_gridSize*sizeof(float2), cudaMemcpyHostToDevice));
}
//----------------------------------------------------------------------------------------------------------------------
void Ocean::initialise(){

    // Assign memory for the frequecy field and heights in the time domain
    h_H0 = (float2*) malloc(m_gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_H0, m_gridSize*sizeof(float2)));
    h_Ht = (float2*)malloc(m_gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_Ht, m_gridSize*sizeof(float2)));
    h_Heights = (float2*)malloc(m_gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_Heights, m_gridSize*sizeof(float2)));
    h_ChopX = (float2*)malloc(m_gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_ChopX, m_gridSize*sizeof(float2)));
    h_ChopY = (float2*)malloc(m_gridSize*sizeof(float2));
    checkCudaErrors(cudaMalloc((void**)&d_ChopY, m_gridSize*sizeof(float2)));


    // Create our frequency domain at time zero
    createH0();

    // create FFT plan
    if(cufftPlan2d(&m_fftPlan, m_resolution, m_resolution, CUFFT_C2C) != CUFFT_SUCCESS) {
        MGlobal::displayError(MString("Cuda: cufftPlan2d CUFFT_C2C failed\n"));
    }
    // update our frequency domain
    update(0.0);
}
//----------------------------------------------------------------------------------------------------------------------
void Ocean::update(double _time){
    updateFrequencyDomain(d_H0, d_Ht, _time, m_resolution);

    cudaThreadSynchronize();

    // Conduct FFT to retrive heights from frequency domain
    cufftExecC2C(m_fftPlan, d_Ht, d_Heights, CUFFT_INVERSE);

    cudaThreadSynchronize();

    // Get the choppiness
    addChoppiness(d_Heights, d_ChopX, d_ChopY, m_resolution, m_windDirection);

}
//----------------------------------------------------------------------------------------------------------------------
float2* Ocean::getHeights(){
    checkCudaErrors(cudaMemcpy(h_Heights, d_Heights, m_gridSize*sizeof(float2), cudaMemcpyDeviceToHost));

    return h_Heights;
}
//----------------------------------------------------------------------------------------------------------------------
float2* Ocean::getChopX(){
    checkCudaErrors(cudaMemcpy(h_ChopX, d_ChopX, m_gridSize*sizeof(float2), cudaMemcpyDeviceToHost));

    return h_ChopX;
}
//----------------------------------------------------------------------------------------------------------------------
float2* Ocean::getChopY(){
    checkCudaErrors(cudaMemcpy(h_ChopY, d_ChopY, m_gridSize*sizeof(float2), cudaMemcpyDeviceToHost));

    return h_ChopY;
}
//----------------------------------------------------------------------------------------------------------------------
void Ocean::setResolution(int _res){
    // Set out new resolution
    m_resolution = _res;
    m_gridSize = m_resolution*m_resolution;

    // Free our memory so we can reinstantiate it at a new size
    free(h_H0);
    cudaFree(d_H0);
    free(h_Ht);
    cudaFree(d_Ht);
    free(h_Heights);
    cudaFree(d_Heights);
    free(h_ChopX);
    cudaFree(d_ChopX);
    free(h_ChopY);
    cudaFree(d_ChopY);

    initialise();
}
//----------------------------------------------------------------------------------------------------------------------
void Ocean::setWindVector(float2 _windVector){
    m_windDirection = _windVector;
}

void Ocean::setWindSpeed(float _speed){
    m_windSpeed = _speed;
    m_L = (m_windSpeed * m_windSpeed) / 9.81;
    m_l = m_L / 1000.0;
}
