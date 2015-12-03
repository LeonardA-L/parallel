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

#include "GPU.cuh"

using namespace std;

void testError(int ok, char* message){
	if(ok != cudaSuccess){
		cerr << message << endl;
	}
}

/***************************************************************************
  	 Prepare the kernel call:
  	 - Transfer the features to the GPU
  	 - Prepare an array for the results, initialized to zero (in parallel on the GPU)
     ***************************************************************************/

  	void preKernel(float *features, float *features_integral,
  		float **_gpuFeatures, float **_gpuFeaturesIntegral, unsigned int **_gpuResult,
  		int16_t w, int16_t h, int16_t w_integral, int16_t h_integral, int16_t noChannels, 
		int numLabels)
  	{
  		cudaError_t ok;
  		int size;

  		//printFreeGPUMem("CUDA Malloc features: ");

		// Allocate GPU memory for the features and transfer
		// them from host memory to GPU memory
		size=noChannels*w*h*sizeof(float);
		ok=cudaMalloc ((void**) _gpuFeatures, size);
		testError(ok, "cudaMalloc error 1");
		ok=cudaMemcpy (*_gpuFeatures, features, size, cudaMemcpyHostToDevice);
  		testError(ok, "cudaMemcpyHostToDevice error 1");

  		size=noChannels*w_integral*h_integral*sizeof(float);
		ok=cudaMalloc ((void**) _gpuFeaturesIntegral, size);
		testError(ok, "cudaMalloc error 2");
		ok=cudaMemcpy (*_gpuFeaturesIntegral, features_integral, size, cudaMemcpyHostToDevice);
  		testError(ok, "cudaMemcpyHostToDevice error 2");
  		
		size=w*h*numLabels*sizeof(unsigned int);
		ok=cudaMalloc ((void**) _gpuResult, size);
		testError(ok, "cudaMalloc error 3");


		//.... KERNEL LAUNCH ICI

  	}

  	/***************************************************************************
  	 After the kernel call:
  	 - Transfer the result back from the GPU to the _CPU
  	 - free the GPU memory related to a single image (features), but not the 
  	   forest!
     ***************************************************************************/

  	void postKernel(float *_gpuFeatures, float *_gpuFeaturesIntegral, unsigned int *_gpuResult,
  		unsigned int *result, int16_t w, int16_t h, int numLabels)
  	{
  		cudaError_t ok;
  		int size;

  		// Copy the results back to host memory
  		size=w*h*numLabels*sizeof(unsigned int);
  		ok=cudaMemcpy (result, _gpuResult, size, cudaMemcpyDeviceToHost);
  		testError(ok, "cudaMemcpyDeviceToHost error 1");

#ifdef GPU_DEBUG_SINGLE_PIXEL
  		std::cerr << "Debug-error code (int)=" << std::dec << (int) *result << "\n";
  		std::cerr << "Return values: ";
  		for (int i=0; i<result[0]; ++i)
  			std::cerr << result[i+1] << " ";
  		std::cerr << "\n";
#endif  		

  		// Free up GPU memory.
  		cudaFree(_gpuFeatures);
  		cudaFree(_gpuFeaturesIntegral);
  		cudaFree(_gpuResult);
  	}



	__device__
  	float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
  		int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h)
    {
        float res = (
        		gpuFeaturesIntegral[y2 + x2*h + channel*w*h] -
                gpuFeaturesIntegral[y2 + x1*h + channel*w*h] -
                gpuFeaturesIntegral[y1 + x2*h + channel*w*h] +
                gpuFeaturesIntegral[y1 + x1*h + channel*w*h]);

        return res;
	}
