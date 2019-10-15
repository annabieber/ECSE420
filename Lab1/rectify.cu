#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>


void rectifyWithCuda(unsigned char* image, unsigned numPixels, unsigned sizeChars, unsigned int threads_per_block);

__global__ void rectify(unsigned char* dev_image, unsigned offset)
{
	// Numbers of blocks in a grid
	unsigned gridWidth = gridDim.x;
	// Fetching index of current block in 1-D (in comparison to the entire grid - equivalent of x + y * width) 
	int currentBlockIndex = blockIdx.x + blockIdx.y * gridWidth;

	// Numbers of threads per block
	unsigned blockWidth = blockDim.x;
	// Fetching index of current thread in 1-D inside the current block (in comparison to the entire grid) 
	int currentThreadIndex = threadIdx.x + currentBlockIndex * blockWidth;

	// In case total numbers of pixel is higher than total numbers of threads available, then will use offset
	// This offset will "pickup" the rest of the work where it was last left.
	int currentPixelStartIndex = currentThreadIndex + offset;

	// 4 bytes per pixel (8 bits per channel R, G, B, Alpha)
	int currentPixelArray = 4 * currentPixelStartIndex;

	// Fetching the current pixel pointer to change its value from the device copy (dev_image)
	// will overwrite the channels value and leave the alpha channel untouched
	unsigned char* currentPixelPointer = dev_image + currentPixelArray * sizeof(char);

	// Checking if Red channel value of this pixel (located at the start of the pointer) is less than 127
	// if so, set it to 127, otherwise, leave it untouched
	if ((int)currentPixelPointer[0] < 127) {
		currentPixelPointer[0] = (unsigned char)127;
	}
	// Similarly, to Green channel
	if ((int)currentPixelPointer[1] < 127) {
		currentPixelPointer[1] = (unsigned char)127;
	}
	// Similarly, to Blue channel
	if ((int)currentPixelPointer[2] < 127) {
		currentPixelPointer[2] = (unsigned char)127;
	}
}

int main(int argc, char* argv[])
{
	// Extracting CLI arguments
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	int threads_per_block = atoi(argv[3]);

	if (threads_per_block > 1024) {
		printf("Error, number of threads per block cannot exceed 1024. Please try another value.");
		return 1;
	}

	unsigned error;
	unsigned char* image;
	unsigned width, height;

	// Decoding and loading the PNG image to the image pointer and setting the width and height values as well
	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	
	unsigned numPixels = width * height;
	// Given that we have 4 channels, we have 4 chars per pixel (8 bytes per channel so 32 bits in total per pixel)
	unsigned sizeChars = numPixels * 4;

	// Method handling device memory setup and launching the kernel
	rectifyWithCuda(image, numPixels, sizeChars, threads_per_block);
	
	// Encoding and saving the new rectified image in a PNG
	lodepng_encode32_file(output_filename, image, width, height);
	
	// Freeing memory
	free(image);
	return 0;
}

void rectifyWithCuda(unsigned char* image, unsigned numPixels, unsigned sizeChars, unsigned int threads_per_block)
{
	// Counter for timing
	double time_elapsed = 0.0;

	// Pointer for device copy of the image
	unsigned char* dev_image;
	// Allocating device memory to it
	cudaMalloc((void**)(&dev_image), sizeChars * sizeof(char));
	// Perfoming the copying
	cudaMemcpy(dev_image, image, sizeChars, cudaMemcpyHostToDevice);

	// Variables used to create the dim3 struct
	int numXBlocks, numYBlocks;

	// Computing the numbers of blocks (x-axis & y-axis)
	numXBlocks = numPixels / threads_per_block;
	numYBlocks = 1;

	// Creating dim3 structs to contain the dimensionality of our GPU that we will be requiring 
	// (Z axis set to 1 for threads and blocks and Y axis set to 1 as well for threads because of hardware possible limitationss)
	dim3 totalThreadsPerBlock(threads_per_block, 1, 1); 
	dim3 totalBlocks(numXBlocks, numYBlocks, 1);

	printf("Using %d blocks (%d in x, %d in y), each with %d threads .\n", numXBlocks*numYBlocks, numXBlocks, numYBlocks, threads_per_block);
	
	// Starting timer
	clock_t star_time = clock();

	// Calling the kernel with an offset value of 0 given that we are at the start (no remainder involved yet)
	rectify <<<totalBlocks, totalThreadsPerBlock >>> (dev_image, 0);

	// Given that the division of numPixels / threads_per_block can have a remainder, then we will be using an "offset"
	// 2^10 - 1 = 1023 is the maximum value of a remainder for decimal numbers divisions so we can use a single block
	int remainder = numPixels - (threads_per_block * numXBlocks * numYBlocks); 
	printf("Remaining pixels %d.\n", remainder);
	
	// Offset can never be negative
	unsigned offset = numPixels - remainder;
	// Launching another kernel to deal with the remaining pixels and giving the offset to locate the starting pixel 
	// of these remaining ones
	rectify <<<1, remainder>>> (dev_image, offset);

	cudaDeviceSynchronize();

	// After all rectification is done, copy the processed image from the device memory to host memory (overwriting)
	cudaMemcpy(image, dev_image, sizeChars, cudaMemcpyDeviceToHost);

	// Clear the device memory allocation
	cudaFree(dev_image);

	// Ending timer
	clock_t end_time = clock();
	time_elapsed += (double)(end_time - star_time) / CLOCKS_PER_SEC;
	printf("Time elapsed for CUDA operation is %f seconds", time_elapsed);

}