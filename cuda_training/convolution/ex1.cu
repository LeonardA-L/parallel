
/*****************************************************************************
CUDA PROGRAMMING EXAMPLES

Authors: Christian Wolf, LIRIS, CNRS, INSA-Lyon 
christian.wolf@liris.cnrs.fr

Changelog:
03.07.15 cw: begin development
*****************************************************************************/

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

using namespace std;

#define blockSize	16
#define TILE_WIDTH	blockSize

clock_t LastProfilingClock=clock();

/***************************************************************************
 Writes profiling output (milli-seconds since last call)
 ***************************************************************************/

extern clock_t LastProfilingClock;

inline float profiling (const char *s, clock_t *whichClock=NULL) 
{
	if (whichClock==NULL)
		whichClock=&LastProfilingClock;

    clock_t newClock=clock();
    float res = (float) (newClock-*whichClock) / (float) CLOCKS_PER_SEC;
    if (s!=NULL)
        std::cerr << "Time: " << s << ": " << res << std::endl; 
    *whichClock = newClock;
    return res;
}

inline float profilingTime (const char *s, time_t *whichClock) 
{
    time_t newTime=time(NULL);
    float res = (float) (newTime-*whichClock);
    if (s!=NULL)
        std::cerr << "Time(real): " << s << ": " << res << std::endl; 
    return res;
}

/* Our stuff */

/***************************************************************************
 USAGE
 ***************************************************************************/

void usage (char *com) 
{
    std::cerr<< "usage: " << com << " <imagename>\n";
    exit(1);
}

/***************************************************************************
 The CPU version
 ***************************************************************************/

void cpuFilter(unsigned char *in, unsigned char * resarr, int rows, int cols){
	for (int y=1; y<rows-1; ++y){
		for (int x=1; x<cols-1; ++x){
			//cout << (int)imarr[x*rows+y] << endl;
			/*int total = 0;
			if(y > 0){	// !TOP
				if(x > 0){	// !LEFT
					total += imarr[(x-1)*rows+(y-1)] * 1;
				}
				total += imarr[(x)*rows+(y-1)] * 2;
				if(x < cols -1){	// !RIGHT
					total += imarr[(x+1)*rows+(y-1)] * 1;
				}
			}
				if(x > 0){	// !LEFT
					total += imarr[(x-1)*rows+(y)] * 2;
				}
				total += imarr[(x)*rows+(y)] * 4;
				if(x < cols -1){	// !RIGHT
					total += imarr[(x+1)*rows+(y)] * 2;
				}
			if(y < rows - 1){
				if(x > 0){	// !LEFT
					total += imarr[(x-1)*rows+(y+1)] * 1;
				}
				total += imarr[(x)*rows+(y+1)] * 2;
				if(x < cols -1){	// !RIGHT
					total += imarr[(x+1)*rows+(y+1)] * 1;
				}
			}
			total /= 16;*/
			int total = (
            4.0*in[x*rows+y] +
            2.0*in[(x-1)*rows+y] +
            2.0*in[(x+2)*rows+y] +
            2.0*in[x*rows+y+1] +
            2.0*in[x*rows+y-1] +
            in[(x-1)*rows+y-1] +
            in[(x-1)*rows+y+1] +
            in[(x+1)*rows+y-1] +
            in[(x+1)*rows+y+1]
            )/16.0;
			
			if(total < 0) total = 0;
			if(total > 255) total = 255;
			//cout << total << endl;
			resarr[x*rows+y] = (unsigned char)total;
		}
	}
}

/***************************************************************************
 The GPU version - the kernel
 ***************************************************************************/

__global__ void onePixel(unsigned char *in, unsigned char *resarr, int * d_cols) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int cols = *d_cols;
		
		int shx = threadIdx.x+1;
		int shy = threadIdx.y+1;
		
		__shared__ unsigned char shIn[(TILE_WIDTH) * (TILE_WIDTH)];
		/*
		int shift = 0;
		
		shIn[shy * (TILE_WIDTH) +	shx + shift] = in[x+y*cols];
		
		// Corners
		if(shx == 1 && shy == 1){
			shIn[(shy-1) * (TILE_WIDTH) + shx-1 + shift] = in[(x-1)+(y-1)*cols];
		}
		else if(shx == TILE_WIDTH && shy == TILE_WIDTH){
			shIn[(shy+1) * (TILE_WIDTH) + shx+1 + shift] = in[(x+1)+(y+1)*cols];
		}
		
		// First Col
		if(shx == 1){
			shIn[shy * (TILE_WIDTH) + shx-1 + shift] = in[(x-1)+(y)*cols];
		}
		else if(shx == blockSize-1){
			shIn[shy * (TILE_WIDTH) + shx+1 + shift] = in[(x+1)+(y)*cols];
		}
		
		// First row
		if(shy == 1){
			shIn[(shy-1) * (TILE_WIDTH) + shx + shift] = in[(x-1)+y*cols];
		}
		else if(shy == blockSize-1){
			shIn[(shy+1) * (TILE_WIDTH) + shx + shift] = in[(x+1)+y*cols];
		}
		*/
		
		__syncthreads();
		
		int total;
		total = (
			4.0*shIn[shx+shy*TILE_WIDTH] +
			2.0*shIn[(shx-1)+shy*TILE_WIDTH] +
			2.0*shIn[(shx+1)+shy*TILE_WIDTH] +
			2.0*shIn[shx+(shy+1)*TILE_WIDTH] +
			2.0*shIn[shx+(shy-1)*TILE_WIDTH] +
			shIn[(shx-1)+(shy-1)*TILE_WIDTH] +
			shIn[(shx-1)+(shy+1)*TILE_WIDTH] +
			shIn[(shx+1)+(shy-1)*TILE_WIDTH] +
			shIn[(shx+1)+(shy+1)*TILE_WIDTH]
			)/16.0;
		
		/*
		 // Non shared memory
			total = (
			4.0*in[x*rows+y] +
			2.0*in[(x-1)*rows+y] +
			2.0*in[(x+2)*rows+y] +
			2.0*in[x*rows+y+1] +
			2.0*in[x*rows+y-1] +
			in[(x-1)*rows+y-1] +
			in[(x-1)*rows+y+1] +
			in[(x+1)*rows+y-1] +
			in[(x+1)*rows+y+1]
			)/16.0;
		*/
		
		if(total < 0) total = 0;
		if(total > 255) total = 255;
		//cout << total << endl;
		resarr[x+y*cols] = (unsigned char)total;
}

__global__ void onePixelLarge(unsigned char *in, unsigned char *resarr, int * d_cols) {
		int x = blockIdx.x * (blockDim.x-2) + threadIdx.x;
		int y = blockIdx.y * (blockDim.y-2) + threadIdx.y;
		int cols = *d_cols;
		
		int shx = threadIdx.x;
		int shy = threadIdx.y;
		
		__shared__ unsigned char shIn[(TILE_WIDTH) * (TILE_WIDTH)];
		
		shIn[shy * (TILE_WIDTH) + shx] = in[x+y*cols];
		
		__syncthreads();
		if(shx == 0 || shx == TILE_WIDTH-1 || shy == 0 || shy == TILE_WIDTH-1 ){
			return;
		}
		int total=0;
		
		total = (
			4.0*shIn[shx+shy*TILE_WIDTH] +
			2.0*shIn[(shx-1)+shy*TILE_WIDTH] +
			2.0*shIn[(shx+1)+shy*TILE_WIDTH] +
			2.0*shIn[shx+(shy+1)*TILE_WIDTH] +
			2.0*shIn[shx+(shy-1)*TILE_WIDTH] +
			shIn[(shx-1)+(shy-1)*TILE_WIDTH] +
			shIn[(shx-1)+(shy+1)*TILE_WIDTH] +
			shIn[(shx+1)+(shy-1)*TILE_WIDTH] +
			shIn[(shx+1)+(shy+1)*TILE_WIDTH]
			)/16.0;
		
		/*
		 // Non shared memory
			total = (
			4.0*in[x*rows+y] +
			2.0*in[(x-1)*rows+y] +
			2.0*in[(x+2)*rows+y] +
			2.0*in[x*rows+y+1] +
			2.0*in[x*rows+y-1] +
			in[(x-1)*rows+y-1] +
			in[(x-1)*rows+y+1] +
			in[(x+1)*rows+y-1] +
			in[(x+1)*rows+y+1]
			)/16.0;
		*/
		
		if(total < 0) total = 0;
		if(total > 255) total = 255;
		//cout << total << endl;
		resarr[x+y*cols] = (unsigned char)total;
}
 /***************************************************************************
 The GPU version - the host code
 ***************************************************************************/

void testError(int ok, char* message){
	if(ok != cudaSuccess){
		cerr << message << endl;
	}
}

void gpuFilter(unsigned char *in, unsigned char * resarr, int rows, int cols){
	long size = sizeof(unsigned char)*cols*rows;
	unsigned char *d_in, *d_out;
	int* d_cols;
	
	cudaError_t ok;
	
	ok=cudaMalloc((void**) &d_in, size);
	testError(ok, "cudaMalloc 1 error");
	ok=cudaMalloc((void**) &d_out, size);
	testError(ok, "cudaMalloc 2 error");
	ok=cudaMalloc((void**) &d_cols, sizeof(int));
	testError(ok, "cudaMalloc 3 error");
	
	ok=cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	testError(ok, "cudaMemcpy 1 error");
	ok=cudaMemcpy(d_cols, &cols, sizeof(int), cudaMemcpyHostToDevice);
	testError(ok, "cudaMemcpy 2 error");
	
	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(rows/(blockSize-2), cols/(blockSize-2));
	
	/*cout << dimBlock.x << " " << dimBlock.y << endl;
	cout << dimGrid.x << " " << dimGrid.y << endl;*/
	
	onePixelLarge<<<dimGrid, dimBlock>>>(d_in, d_out, d_cols);
	ok = cudaGetLastError();
	//cerr << "CUDA Status :"<< cudaGetErrorString(ok) << endl;
	testError(ok, "error kernel launch");
	
	//cout << &resarr << endl;
	
	ok=cudaMemcpy(resarr, d_out, size, cudaMemcpyDeviceToHost);
	testError(ok, "cudaMemcpy deviceToHost error");
	
	ok=cudaFree(d_in);
	testError(ok, "cudaFree 1 error");
	ok=cudaFree(d_out);
	testError(ok, "cudaFree 2 error");
	ok=cudaFree(d_cols);
	testError(ok, "cudaFree 3 error");
	
}
	

/***************************************************************************
 Main program
 ***************************************************************************/


int main (int argc, char **argv)
{
	int c;
	// Argument processing
    while ((c =	getopt (argc, argv,	"h")) != EOF) 
    {
		switch (c) {

			case 'h':
				usage(*argv);
				break;
	
			case '?':
				usage (*argv);
				std::cerr << "\n" << "*** Problem parsing the options!\n\n";
				exit (1);
		}
	}	

    int requiredArgs=2;

	if (argc-optind!=requiredArgs) 
    {
        usage (*argv);
		exit (1);
	}
	char *inputfname=argv[optind];
	char *outputfname=argv[optind+1];

	cv::Mat im = cv::imread(inputfname,-1);
	if (!im.data)
	{
		std::cerr << "*** Cannot load image: " << inputfname << "\n";
		exit(1);
	}
	std::cout << "=====================================================\n"
		<< "Loaded image of size " << im.cols << "x" << im.rows << ".\n";
	cv::Mat result (im.rows, im.cols, CV_8U);

	// Copy the cv::Mat into a linear array
	unsigned char *imarr = new unsigned char [im.cols*im.rows];
	for (int y=0; y<im.rows; ++y)
	for (int x=0; x<im.cols; ++x)
		imarr[x+y*im.cols] = im.at<unsigned char>(y,x);
	unsigned char *resarr = new unsigned char [im.cols*im.rows];
	profiling (NULL);
	int nMax = 1000;

	// Each version is run a 100 times to have 
	// a better idea on run time
	/*
	for (int i=0; i<nMax; ++i)
		cpuFilter(imarr, resarr, im.rows, im.cols);

	profiling ("CPU version");
	*/
	for (int i=0; i<nMax; ++i)
		gpuFilter(imarr, resarr, im.rows, im.cols);

	profiling ("GPU version");

	// Copy the linear array back to the cv::Mat
	for (int y=0; y<im.rows; ++y)
	for (int x=0; x<im.cols; ++x)
		result.at<unsigned char>(y,x) = resarr[x+y*im.cols];

	imwrite (outputfname, result);

    std::cout << "Program terminated correctly.\n";
    return 0;
}

