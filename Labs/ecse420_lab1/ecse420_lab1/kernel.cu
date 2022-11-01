#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "b_32.h"
#include "a_32.h"
#include "a_10.h"
#include "x_32.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"


#define mat_elem(a, y, x, n) (a + ((y) * (n) + (x)))
//size of the matrix and the vectors
#define N 32

//struct for a matrix with height and width and the elements 
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

//input full arrays a and b, diagonale value, the max_row and size n
void swap_row(float* a, float* b, int r1, int r2)
{
	//create variables for later
	float tmp, * p1, * p2;

	//if the diagonal is the max then we're not swapping cause they will be the same
	if (r1 == r2) return;
	//swap the elements of each row 
	for (int i = 0; i < N; i++) {
		p1 = mat_elem(a, r1, i, N);
		p2 = mat_elem(a, r2, i, N);
		tmp = *p1, * p1 = *p2, * p2 = tmp;
	}
	tmp = b[r1], b[r1] = b[r2], b[r2] = tmp;
}

//input two full arrays a and b, one empty array x and size of x and b
void gauss_eliminate(float a[][N], float b[], float* x)
{
	//create variables for later
	int i, j, col, row, max_row, dia;
	float max, tmp;

	//iterate over the matrix diagonaly, n is the size of x and y
	//iterate by row/diagonal
	for (dia = 0; dia < N; dia++) {

		//max_row takes on the current row/diagonal
		//max are the values of the diagonal at every step i.e. (0,0), (1,1), ... , (n,n)
		max_row = dia, max = a[dia][dia];

		//iterate over the rows
		for (row = dia + 1; row < N; row++)
			//if the aboslute value of A at the row and column is bigger than the diagonal 
			//want to check if the values on the row is bigger than the value where the 1 for the identity matrix 
			if ((tmp = fabs(a[row][dia])) > max)
				//swap max_row with row and max with temp
				max_row = row, max = tmp;

		//swap rows - see method above
		swap_row(*a, b, dia, max_row);

		//iterate over the rows again 
		for (row = dia + 1; row < N; row++) {
			//divide the row and dia by dia, dia (position of the 1 of the identity matrix) and store it in a temp variable
			tmp = a[row][dia] / a[dia][dia];

			//iterate over the columns in the matrix
			for (col = dia + 1; col < N; col++)
				//calculate the new value for at every column and row
				a[row][col] -= tmp * a[dia][col];
			a[row][dia] = 0;

			//calculate the new value of be for each row
			b[row] -= tmp * b[dia];
		}
	}

	double temp;
	//matrix multiplication -- need to parallelize
	for (int row = N - 1; row >= 0; row--) {
		//store the value of b[i] in a temp variable
		temp = b[row];
		//iterate over the column in A and multiply by the value of b
		for (int j = N - 1; j > row; j--)
			//calculate x*a to, store and substract in temp 
			temp -= x[j] * a[row][j];
		//divide by A 
		x[row] = temp / a[row][row];
	}
}

__global__ void multiply(Matrix dev_A, Matrix dev_x, Matrix dev_b)
{
	// Each thread computes one element of B
	// by accumulating results into B_value
	float b_value = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < dev_A.width; ++e)
		//calculate the value for each element and then sum it with the previous calculated values
		b_value += dev_A.elements[row * dev_A.width + e]
		* dev_x.elements[e * dev_x.width + col];
	dev_b.elements[row * dev_b.width + col] = b_value;
}


int main(void)
{
	float x_10[N];
	int i;
	//If running the matrix inversion and multiplication together run the following	
	//gauss_eliminate(A_10, b_10, x_10);
	//for (i = 0; i < N; i++)
		//printf("%f\n", x_10);


	//if running only the multiplication part run the following instruction.

	Matrix A;
	A.height = N;
	A.width = N;
	//A_32 is a 1D 32*32 flattened array 
	A.elements = A_32;

	float b_elem[N] = {
			 1, 2, 3, 4, 5, 6, 7, 8, 9, 1,
			 2, 3, 4, 5, 6, 7, 8, 9, 1, 2,
			 3, 4, 5, 6, 7, 8, 9, 1, 2, 3,
			 4, 5
	};
	Matrix B;
	B.height = N;
	B.width = 1;
	B.elements = b_elem;

	Matrix x;
	x.height = N;
	x.width = 1;
	x.elements = X_32;

	//create a matrix for A and allocate memory
	Matrix dev_A32;
	dev_A32.height = A.height; dev_A32.width = A.width;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&dev_A32.elements, size);
	cudaMemcpy(dev_A32.elements, A.elements, size, cudaMemcpyHostToDevice);

	//create vector for x and allocate memory
	Matrix dev_x;
	dev_x.height = x.height; dev_x.width = x.width;
	size = x.width * x.height * sizeof(float);
	cudaMalloc(&dev_x.elements, size);
	cudaMemcpy(dev_x.elements, x.elements, size, cudaMemcpyHostToDevice);

	//create vector for b and allocate memory
	Matrix dev_b;
	dev_b.width = B.width; dev_b.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&dev_b.elements, size);

	//create a grid with blocks for threads
	dim3 dimBlock(16, 16);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	//matrix multiplication with blocks and threads 
	multiply <<<dimGrid, dimBlock >> > (dev_A32, dev_x, dev_b);

	//copy B
	cudaMemcpy(B.elements, dev_b.elements, size, cudaMemcpyDeviceToHost);

	//display the elements in B
	for (i = 0; i < N; i++)
		printf("%f\n", B.elements[i]);

	return 0;
}