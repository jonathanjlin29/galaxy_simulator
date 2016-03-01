#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

#include </usr/local/cuda/include/cuda_runtime.h>
#include "cuda_step.h"

using namespace std;

#define CUDA_BLOCK 32

__global__ void calculateStepParticles(double *forceXd, double *forceYd, double *forceZd, double *posXd, double *posYd, double *posZd,
				       double *massd, double *velXd, double *velYd, double *velZd, double h, int sizeX, int numParticlesPadded, double e2) {
   int tNdx = blockIdx.x * CUDA_BLOCK + threadIdx.x;

   double forcex = 0, forcey = 0, forcez = 0;
   for(int i = 0; i < numParticlesPadded; i++) {
      if(tNdx != i) {
	      double rijx = posXd[i] - posXd[tNdx];
	      double rijy = posYd[i] - posYd[tNdx];
	      double rijz = posZd[i] - posZd[tNdx];
	      double normalize = pow(rijx*rijx + rijy*rijy + rijz*rijz, 0.5);
	      double rsquared = normalize * normalize;
	      double numerator = massd[tNdx] * massd[i];
	      double denominator = pow(rsquared + e2, 3.0/2);
	      double multiplier = numerator/denominator;
    
	      forcex += multiplier * rijx;
	      forcey += multiplier * rijy;
	      forcez += multiplier * rijz;
      }
   }
   

   forceXd[tNdx] = forcex;
   forceYd[tNdx] = forcey;
   forceZd[tNdx] = forcez;

   velXd[tNdx] += (h * 1.0/massd[tNdx]) * forceXd[tNdx];
   velYd[tNdx] += (h * 1.0/massd[tNdx]) * forceYd[tNdx];
   velZd[tNdx] += (h * 1.0/massd[tNdx]) * forceZd[tNdx];

   posXd[tNdx] += (h * velXd[tNdx]);
   posYd[tNdx] += (h * velYd[tNdx]);
   posZd[tNdx] += (h * velZd[tNdx]);
}

void stepParticles(vector<double> &positionx, vector<double> &positiony , vector<double> &positionz, vector<double> &masses,
		   vector<double> &velocityx, vector<double> &velocityy, vector<double> &velocityz, double h, double *t, double e2)
{
    int sizeX = positionx.size();
    int sizeY = positiony.size();
    int sizeZ = positionz.size();

    int numParticlesPadded = (int) (ceil(1.0 * sizeX / CUDA_BLOCK)) * CUDA_BLOCK;

    // setting up device variables
    double *forceXd = NULL, *forceYd = NULL, *forceZd = NULL;
    double *posXd = NULL, *posYd = NULL, *posZd = NULL;
    double *massd = NULL;
    double *velXd = NULL, *velYd = NULL, *velZd = NULL;

    // cudaMalloc with padding
    cudaError_t cudaerror = cudaMalloc((void **) &forceXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &forceYd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &forceZd, sizeof(double) * numParticlesPadded);
    cudaerror = cudaMalloc((void **) &posXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &posYd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &posZd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &massd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &velXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &velYd, sizeof(double) * numParticlesPadded);
    cudaerror = cudaMalloc((void **) &velZd, sizeof(double) * numParticlesPadded);

    // copy data from host to device in preparation for calculation
    cudaerror = cudaMemcpy(posXd, &positionx[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(posYd, &positiony[0], sizeof(double) * sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(posZd, &positionz[0], sizeof(double) * sizeZ, cudaMemcpyHostToDevice);
    cudaMemcpy(massd, &masses[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(velXd, &velocityx[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(velYd, &velocityy[0], sizeof(double) * sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(velZd, &velocityz[0], sizeof(double) * sizeZ, cudaMemcpyHostToDevice);

    dim3 dimBlock(CUDA_BLOCK, 1);
    dim3 dimGrid(numParticlesPadded / CUDA_BLOCK, 1);

    // kernel call
    calculateStepParticles<<<dimGrid, dimBlock>>>(forceXd, forceYd, forceZd, posXd, posYd, posZd, massd, velXd, velYd, velZd, h, sizeX, numParticlesPadded, e2);

    // copy data back from device to host
    cudaMemcpy(&positionx[0], posXd, sizeof(double) * sizeX, cudaMemcpyDeviceToHost);
    cudaMemcpy(&positiony[0], posYd, sizeof(double) * sizeY, cudaMemcpyDeviceToHost);
    cudaMemcpy(&positionz[0], posZd, sizeof(double) * sizeZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(&velocityx[0], velXd, sizeof(double) * sizeX, cudaMemcpyDeviceToHost);
    cudaMemcpy(&velocityy[0], velYd, sizeof(double) * sizeY, cudaMemcpyDeviceToHost);
    cudaMemcpy(&velocityz[0], velZd, sizeof(double) * sizeZ, cudaMemcpyDeviceToHost);

    // free cudaMalloc
    cudaFree(forceXd);
    cudaFree(forceYd);
    cudaFree(forceZd);
    cudaFree(posXd);
    cudaFree(posYd);
    cudaFree(posZd);
    cudaFree(massd);
    cudaFree(velXd);
    cudaFree(velYd);
    cudaFree(velZd);

    *t += h;
}
