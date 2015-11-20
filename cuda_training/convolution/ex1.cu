
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

void cpuFilter(unsigned char *imarr, unsigned char *, int rows, int cols){
	for (int y=0; y<rows; ++y){
		for (int x=0; x<cols; ++x){
			//cout << (int)imarr[x*rows+y] << endl;
			int total = 0;
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
			
			total /= 16;
			//cout << total << endl;
			imarr[x*rows+y] = (char)total;
		}
	}
}

/***************************************************************************
 The GPU version - the kernel
 ***************************************************************************/

// ...

 /***************************************************************************
 The GPU version - the host code
 ***************************************************************************/

// ...
	

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
		imarr[x*im.rows+y] = im.at<unsigned char>(y,x);
	unsigned char *resarr = new unsigned char [im.cols*im.rows];

	profiling (NULL);


	// Each version is run a 100 times to have 
	// a better idea on run time
	
	for (int i=0; i<100; ++i)
		cpuFilter(imarr, resarr, im.rows, im.cols);

	profiling ("CPU version");

	//for (int i=0; i<100; ++i)
	//	gpuFilter(imarr, resarr, im.rows, im.cols);

	profiling ("GPU version");

	// Copy the linear array back to the cv::Mat
	for (int y=0; y<im.rows; ++y)
	for (int x=0; x<im.cols; ++x)
		result.at<unsigned char>(y,x) = resarr[x*im.rows+y];

	imwrite (outputfname, result);

    std::cout << "Program terminated correctly.\n";
    return 0;
}

