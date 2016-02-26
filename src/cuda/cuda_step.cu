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
      double *massd, double *velXd, double *velYd, double *velZd, int h, int sizeX, int numParticlesPadded) {
   int tNdx = blockIdx.x * CUDA_BLOCK + threadIdx.x;    // <-- might
							// need to
							// check this

   double e2 = 1e-4;
   //double h = 1.0;


   /* Shared Memory Approach
   __shared__ double posXds[numParticlesPadded];
   __shared__ double posYds[numParticlesPadded];
   __shared__ double posZds[numParticlesPadded];
   __shared__ double massds[numParticlesPadded];
   __shared__ double velXds[numParticlesPadded];
   __shared__ double velYds[numParticlesPadded];
   __shared__ double velZds[numParticlesPadded];

   // branch divergence?
   // attempting to copy all data
   if(tNdx < CUDA_BLOCK) {
      for(int i = 0; i < numParticlesPadded / CUDA_BLOCK; i++) {

      }
   }
   */

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
		   vector<double> &velocityx, vector<double> &velocityy, vector<double> &velocityz, int h, int t)
{
    //
    // IMPLEMENT ME
    // PreRequisite: particles have been loaded into global variable
    // vector< shared_ptr<Particle> > particles;
    //vector <Vector3d> forces;

//    for (int i = 0; i < particles.size(); i++) {
//        Vector3d force(0.0, 0.0, 0.0);
//        for (int j = 0; j < particles.size(); j++) {
//            shared_ptr<Particle> partI = particles.at(i);
//            shared_ptr<Particle> partJ = particles.at(j);
//            if(j != i) {
//                Eigen::Vector3d rij = partJ->getPosition() - partI->getPosition();
//                double rsquared = pow(rij.norm(), 2);
//                double numerator = partI->getMass() * partJ->getMass();
//                double denominator = pow(rsquared + e2, 3.0/2);
//                force += (numerator * rij)/denominator;
//            }
//        }
//        forces.push_back(force);
//    }

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
    cudaMalloc((void **) &forceXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &forceYd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &forceZd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &posXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &posYd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &posZd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &massd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &velXd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &velYd, sizeof(double) * numParticlesPadded);
    cudaMalloc((void **) &velZd, sizeof(double) * numParticlesPadded);

    // copy data from host to device in preparation for calculation
    cudaMemcpy(posXd, &positionx[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(posYd, &positiony[0], sizeof(double) * sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(posZd, &positionz[0], sizeof(double) * sizeZ, cudaMemcpyHostToDevice);
    cudaMemcpy(massd, &masses[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(velXd, &velocityx[0], sizeof(double) * sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(velYd, &velocityy[0], sizeof(double) * sizeY, cudaMemcpyHostToDevice);
    cudaMemcpy(velZd, &velocityz[0], sizeof(double) * sizeZ, cudaMemcpyHostToDevice);

    dim3 dimBlock(CUDA_BLOCK, 1);
    dim3 dimGrid(numParticlesPadded / CUDA_BLOCK, 1);

    // kernel call
    //MatMulShared<<<dimGrid, dimBlock>>>(matrixAd, matrixBd, resultMatrixd, matrixANewRow, matrixBNewCol, matrixANewCol);
    calculateStepParticles<<<dimGrid, dimBlock>>>(forceXd, forceYd, forceZd, posXd, posYd, posZd, massd, velXd, velYd, velZd, h, sizeX, numParticlesPadded);

    // copy data back from device to host
    cudaMemcpy(&positionx[0], posXd, sizeof(double) * sizeX, cudaMemcpyDeviceToHost);
    cudaMemcpy(&positiony[0], posYd, sizeof(double) * sizeY, cudaMemcpyDeviceToHost);
    cudaMemcpy(&positionz[0], posZd, sizeof(double) * sizeZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(&velocityx[0], velXd, sizeof(double) * sizeX, cudaMemcpyDeviceToHost);
    cudaMemcpy(&velocityy[0], velYd, sizeof(double) * sizeY, cudaMemcpyDeviceToHost);
//    cudaMemcpy(&velocityz[0]. velZd, sizeof(double) * sizeZ, cudaMemcpyDeviceToHost);

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

/* OPENMP Implementation
    double *forceX = (double *)malloc(sizeof(double) * positionx.size());
    double *forceY = (double *)malloc(sizeof(double) * positionx.size());
    double *forceZ = (double *)malloc(sizeof(double) * positionx.size());

    double *posX = &positionx[0];
    double *posY = &positiony[0];
    double *posZ = &positionz[0];

    double *mass = &masses[0];

    int sizeX = positionx.size();
    int sizeY = positiony.size();
    int sizeZ = positionz.size();

    double *velx = &velocityx[0];
    double *vely = &velocityy[0];
    double *velz = &velocityz[0];

   #pragma omp parallel for
   for (int i = 0; i < sizeX; i++ ) {
      double forcex = 0, forcey = 0, forcez = 0;
	   #pragma simd
      for(int j = 0; j < sizeX; j++ ) {
	      if ( j != i ) {
	    double rijx = posX[j] - posX[i];
	    double rijy = posY[j] - posY[i];
	    double rijz = posZ[j] - posZ[i];
	    double normalize = pow(rijx*rijx + rijy*rijy + rijz*rijz, 0.5);
	    double rsquared = normalize * normalize;
	    double numerator = mass[i] * mass[j];
	   	   double denominator = pow(rsquared + e2, 3.0/2);
		      double multiplier = numerator/denominator;
		      forcex += multiplier * rijx;
	   	   forcey += multiplier * rijy;
	   	   forcez += multiplier * rijz;
	      }
      }
	   //forceX.push_back(forcex);
	   //forceY.push_back(forcey);
	   //forceZ.push_back(forcez);
	   forceX[i] = forcex;
	   forceY[i] = forcey;
	   forceZ[i] = forcez;
   }

//    for(int i = 0; i < particles.size(); i++) {
//        particles.at(i)->updateParticleVelocity(forces.at(i), h);
//        particles.at(i)->updateParticlePosition(forces.at(i), h);
//    }
   #pragma omp parallel for
   for (int i = 0; i < positionx.size(); i++) {
      velx[i] += (h * 1.0/mass[i]) * forceX[i];
      vely[i] += (h * 1.0/mass[i]) * forceY[i];
      velz[i] += (h * 1.0/mass[i]) * forceZ[i];

      posX[i] += (h * velx[i]);
      posY[i] += (h * vely[i]);
      posZ[i] += (h * velz[i]);
   }
*/
   t += h;
}
