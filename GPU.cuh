/* C stuff */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

// Open-CV for the vision stuff
#include <opencv2/opencv.hpp>

/* Cuda stuff */
#include <cuda_runtime_api.h>
#include <cuda.h>


void preKernel(float *features, float *features_integral,
  		float **_gpuFeatures, float **_gpuFeaturesIntegral, unsigned int **_gpuResult,
  		int16_t w, int16_t h, int16_t w_integral, int16_t h_integral, int16_t noChannels, 
		int numLabels);
		
void postKernel(float *_gpuFeatures, float *_gpuFeaturesIntegral, unsigned int *_gpuResult,
  		unsigned int *result, int16_t w, int16_t h, int numLabels);	
  		
__device__ float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
  		int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h);
