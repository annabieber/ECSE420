#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>


void poolWithCuda(unsigned char* image_in, unsigned char* image_out, unsigned numPixels_in, unsigned numPixels_out, unsigned sizeChars_in, unsigned sizeChars_out, unsigned width, unsigned int threads_per_block);


__global__ void pool(unsigned char* dev_image_in, unsigned char* dev_image_out, unsigned width_in, unsigned offset)
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

	// Pointer with which we will navigate through out the 2x2 pool window
	unsigned char* currentPixelPointer;

	// Variable that will hold the maximum value for a channel (R, G or B)
	unsigned maxChannelValue;

	// Variable that will hold the current value of a channel
	unsigned channelValue;

	// We will be traversing channel by channel throughout the entire image
	// to fetch the maximum of each channel even if it is not in the same image inside a pool
	for (int channelIndex = 0; channelIndex < 4; channelIndex++) {
		// For RGB channel
		if (channelIndex < 3) {
			// At the start of each channel sweep, initializing the max value to 0
			maxChannelValue = 0;
			// Starting of iterating through the Y-axis of the pool to use the (x + y * width formula)
			for (int pool_y_index = 0; pool_y_index < 2; pool_y_index++) {
				// Fetching the X-axis index in the 2x2 pool window
				unsigned blockOffset = 4 * 2 * (width_in * (currentPixelStartIndex / (width_in / 2)) + (currentPixelStartIndex % (width_in / 2)));
				// Pointing to the current pixel in the 2x2 pool window by adding the X-axis index and the y * width equivalent inside this pool
				// to the input image starting address
				currentPixelPointer = dev_image_in + blockOffset + 4 * width_in * pool_y_index;
				// Now iterating through the X-axis of the 2x2 pool window
				for (int pool_x_index = 0; pool_x_index < 2; pool_x_index++) {
					// We know that a pixel contains 4 chars (one for each channel)
					int currentPoolPixelIndex = 4 * pool_x_index;
					// We can point to next pixel in the X-axis by simply skipping the current pixel in the x-axis of the pool window
					currentPixelPointer += currentPoolPixelIndex;
					// Fetching the current channel value of the current pixel
					channelValue = (int)currentPixelPointer[channelIndex];
					// If current vlaue is higher than the previous maximu, replace it
					if (channelValue > maxChannelValue) {
						maxChannelValue = channelValue;
					}
				}
			}
			// Once the current channel maximum of a pool has been found
			channelValue = (unsigned char) maxChannelValue;
		}
		// For Alpha channel, set the maximum (255)
		else {
			channelValue = (unsigned char) 255;
		}
		// After assessing the maximum value of a channel of a 2x2 pool, set it in the output image
		dev_image_out[4*currentPixelStartIndex + channelIndex] = channelValue;
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
	unsigned char* inputImage,* outputImage;
	unsigned width, height, width_out, height_out;


	// Decoding and loading the PNG image to the image pointer and setting the width and height values as well
	error = lodepng_decode32_file(&inputImage, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));


	width_out = width / 2;
	height_out = height / 2;
	unsigned numPixels_in = width * height;
	unsigned numPixels_out = width_out * height_out;

	// Total size of output image is going to be the output number of pixels times the size of a char times 4
	// given that each pixels is 32-bits so 4 chars (4 channels of 8 bits each)
	outputImage = (unsigned char*)malloc(4 * numPixels_out * sizeof(char));


	// Given that we have 4 channels, we have 4 chars per pixel (8 bytes per channel so 32 bits in total per pixel)
	unsigned sizeChars_in = numPixels_in * 4;
	unsigned sizeChars_out = numPixels_out * 4;

	// Method handling device memory setup and launching the kernel
	poolWithCuda(inputImage, outputImage, numPixels_in, numPixels_out, sizeChars_in, sizeChars_out, width, threads_per_block);

	// Encoding and saving the new max pooled image in a PNG
	lodepng_encode32_file(output_filename, outputImage, width_out, height_out);

	// Freeing memory
	free(inputImage);
	free(outputImage);
	return 0;
}

void poolWithCuda(unsigned char* image_in, unsigned char* image_out, unsigned numPixels_in, unsigned numPixels_out, unsigned sizeChars_in, unsigned sizeChars_out, unsigned width, unsigned int threads_per_block)
{
	// Counter for timing
	double time_elapsed = 0.0;

	// Pointer for device copy of the image
	unsigned char* dev_image_in, * dev_image_out;
	// Allocating device memory to it
	cudaMalloc((void**)(&dev_image_in), sizeChars_in * sizeof(char));
	cudaMalloc((void**)(&dev_image_out), sizeChars_out * sizeof(char));

	// Perfoming the copying
	cudaMemcpy(dev_image_in, image_in, sizeChars_in, cudaMemcpyHostToDevice);

	// Variables used to create the dim3 struct
	int numXBlocks, numYBlocks;

	// Computing the numbers of blocks (x-axis & y-axis)
	numXBlocks = numPixels_out / threads_per_block;
	numYBlocks = 1;

	// Creating dim3 structs to contain the dimensionality of our GPU that we will be requiring 
	// (Z axis set to 1 for threads and blocks and Y axis set to 1 as well for threads because of hardware possible limitationss)
	dim3 totalThreadsPerBlock(threads_per_block, 1, 1);
	dim3 totalBlocks(numXBlocks, numYBlocks, 1);

	printf("Using %d blocks (%d in x, %d in y), each with %d threads .\n", numXBlocks * numYBlocks, numXBlocks, numYBlocks, threads_per_block);

	// Starting timer
	clock_t star_time = clock();

	// Calling the kernel with an offset value of 0 given that we are at the start (no remainder involved yet)
	pool <<<totalBlocks, totalThreadsPerBlock >> > (dev_image_in, dev_image_out, width, 0);
	// Given that the division of numPixels_out / threads_per_block can have a remainder, then we will be using an "offset"
	// 2^10 - 1 = 1023 is the maximum value of a remainder for decimal numbers divisions so we can use a single block
	int remainder = numPixels_in - (threads_per_block * numXBlocks * numYBlocks);
	printf("Remaining pixels %d.\n", remainder);
	// Offset can never be negative
	unsigned offset = numPixels_out - remainder;
	// Launching another kernel to deal with the remaining pixels and giving the offset to locate the starting pixel 
	// of these remaining ones
	pool <<<1, remainder >>> (dev_image_in, dev_image_out, width, offset);
	cudaDeviceSynchronize();
	// After all pooling is done, copy the processed image from the device memory to host memory (overwriting)
	cudaMemcpy(image_out, dev_image_out, sizeChars_out, cudaMemcpyDeviceToHost);
	// Clear the device memory allocation
	cudaFree(dev_image_in);
	cudaFree(dev_image_out);
	// Ending timer
	clock_t end_time = clock();
	time_elapsed += (double)(end_time - star_time) / CLOCKS_PER_SEC;
	printf("Time elapsed for CUDA operation is %f seconds", time_elapsed);
}