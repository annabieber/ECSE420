#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define mat_elem(a, y, x, n) (a + ((y) * (n) + (x)))
#define N 6

//input full arrays a and b, diagonale value, the max_row and size n
void swap_row(double* a, double* b, int r1, int r2)
{
	//create variables for later
	double tmp, * p1, * p2;
	int i;

	//if the diagonal is the max then we're not swapping cause they will be the same
	if (r1 == r2) return;
	//swap the elements of each row 
	for (i = 0; i < N; i++) {
		p1 = mat_elem(a, r1, i, N);
		p2 = mat_elem(a, r2, i, N);
		tmp = *p1, * p1 = *p2, * p2 = tmp;
	}
	tmp = b[r1], b[r1] = b[r2], b[r2] = tmp;
}

void matrix_multiplication(double a[N][N], double b[], double* x) {
	double temp;
	//matrix multiplication -- need to parallelize
	for (int row = N - 1; row >= 0; row--) {
		//store the value of b[i] in a temp variable
		temp = b[row];
		//iterate over the column in A and multiply by the value of b
		for (int j = N - 1; j > row; j--)
			temp -= x[j] * a[row][j];
		//divide by A ? 
		x[row] = temp / a[row][row];
	}

}

//input two full arrays a and b, one empty array x and size of x and b
void gauss_eliminate(double a[N][N], double b[], double* x)
{
	//create variables for later
	int i, j, col, row, max_row, dia;
	double max, tmp;

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
		swap_row(a, b, dia, max_row, N);

		//iterate over the rows again 
		for (row = dia + 1; row < N; row++) {
			//divide the row and dia by dia, dia (position of the 1 of the identity matrix) and store it in a temp variable
			tmp = a[row][dia] / a[dia][dia];

			//iterate over the columns in the matrix
			for (col = dia + 1; col < N; col++)

				a[row][col] -= tmp * a[dia][col];
			a[row][dia] = 0;
			b[row] -= tmp * b[dia];
		}
	}

	//where we need to use threads
	matrix_multiplication(a, b, x, N);
}

int main(void)
{
	double a[N][N] = {
		{1.00, 0.00, 0.00,  0.00,  0.00, 0.00},
	{1.00, 0.63, 0.39,  0.25,  0.16, 0.10},
	{1.00, 1.26, 1.58,  1.98,  2.49, 3.13},
	{1.00, 1.88, 3.55,  6.70, 12.62, 23.80},
	{1.00, 2.51, 6.32, 15.88, 39.90, 100.28},
	{1.00, 3.14, 9.87, 31.01, 97.41, 306.02}
	};
	double b[N] = { -0.01, 0.61, 0.91, 0.99, 0.60, 0.02 };
	double x[N];
	int i;

	gauss_eliminate(a, b, x);

	for (i = 0; i < N; i++)
		printf("%g\n", x[i]);

	return 0;
}
