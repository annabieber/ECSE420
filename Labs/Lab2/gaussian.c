
#include 
  
   
int n;
float a[10][11];
void forwardSubstitution() {
	int i, j, k, max;
	float t;
	for (i = 0; i < n; ++i) {
		max = i;
		for (j = i + 1; j < n; ++j)
			if (a[j][i] > a[max][i])
				max = j;
		
		for (j = 0; j < n + 1; ++j) {
			t = a[max][j];
			a[max][j] = a[i][j];
			a[i][j] = t;
		}
		
		for (j = n; j >= i; --j)
			for (k = i + 1; k < n; ++k)
				a[k][j] -= a[k][i]/a[i][i] * a[i][j];
/*		for (k = 0; k < n; ++k) {
			for (j = 0; j < n + 1; ++j)
				printf("%.2f\t", a[k][j]);
			printf("\n");
		}*/
	}
}
void reverseElimination() {
	int i, j;
	for (i = n - 1; i >= 0; --i) {
		a[i][n] = a[i][n] / a[i][i];
		a[i][i] = 1;
		for (j = i - 1; j >= 0; --j) {
			a[j][n] -= a[j][i] * a[i][n];
			a[j][i] = 0;
		}
	}
}
void gauss() {
	int i, j;
	forwardSubstitution();
	reverseElimination();
	
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n + 1; ++j)
			printf("%.2f\t", a[i][j]);
		printf("\n");
	}
}
int main(int argc, char *argv[]) {
	int i, j;
	FILE *fin = fopen("gauss.in", "r");
	fscanf(fin, "%d", &n);
	for (i = 0; i < n; ++i)
		for (j = 0; j < n + 1; ++j)
			fscanf(fin, "%f", &a[i][j]);
	fclose(fin);
	
	gauss();
	
	return 0;
}