#ifndef CUDA_STEP
#define CUDA_STEP

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

void stepParticles(vector<double> &positionx, vector<double> &positiony ,
		   vector<double> &positionz, vector<double> &masses,
		   vector<double> &velocityx, vector<double> &velocityy,
		   vector<double> &velocityz, double h, double *t, double e2);
#endif
