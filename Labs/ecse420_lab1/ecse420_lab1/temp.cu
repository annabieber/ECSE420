//#include<stdio.h>
//
//#define N 2
//#include "b_32.h"
//#include "a_32.h"
//
//
//void matrix_multiplication(float a[][N], float b, float* x) {
//	/* this loop is for backward substitution*/
//
//	float sum; 
//
//	
//}
//
//
//
//
//int main()
//{
//	int i, j, k, n;
//	//float A[20][20], c, x[10], sum = 0.0;
//	//printf("\nEnter the order of matrix: ");
//	//scanf("%d", &n);
//	//printf("\nEnter the elements of augmented matrix row-wise:\n\n");
//	/*for (i = 1; i <= n; i++)
//	{
//		for (j = 1; j <= (n + 1); j++)
//		{
//			printf("A[%d][%d] : ", i, j);
//			scanf("%f", &A[i][j]);
//		}
//	}*/ 
//
//	float A[][N] = { {1, 2}, {18, 20} };
//	float b[] = { 4, 5 };
//	float x[N];
//
//
//	for (j = 1; j <= n; j++) /* loop for the generation of upper triangular matrix*/
//	{
//		for (i = 1; i <= n; i++)
//		{
//			if (i > j)
//			{
//				c = A[i][j] / A[j][j];
//				for (k = 1; k <= n + 1; k++)
//				{
//					A[i][k] = A[i][k] - c * A[j][k];
//				}
//			}
//		}
//	}
//	x[n] = A[n][n + 1] / A[n][n];
//
//
//	for (i = n - 1; i >= 1; i--)
//	{
//		sum = 0;
//		for (j = i + 1; j <= n; j++)
//		{
//			sum = sum + A[i][j] * x[j];
//		}
//		x[i] = (A[i][n + 1] - sum) / A[i][i];
//	}
//	
//	printf("\nThe solution is: \n");
//	for (i = 1; i <= n; i++)
//	{
//		printf("\nx%d=%f\t", i, x[i]); /* x1, x2, x3 are the required solutions*/
//	}
//	return(0);
//}