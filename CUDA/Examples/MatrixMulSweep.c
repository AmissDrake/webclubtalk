#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

void multiplyMatrices(int **A, int **B, int **result, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                int64_t temp = (int64_t)A[i][k] * B[k][j];
                result[i][j] += (int)temp;
            }
        }
    }
}

int **allocateMatrix(int N) {
    int **matrix = (int **)malloc(N * sizeof(int *));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        matrix[i] = (int *)malloc(N * sizeof(int));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
    }
    return matrix;
}

void freeMatrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initializeMatrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10;  // Random values between 0 and 9
        }
    }
}

int main() {

    int sizes[6] = {1,10,100,1000,1500,1800};
    
    clock_t start, end;
    double cpu_time_used;

    for(int idx = 0; idx<sizeof(sizes)/sizeof(sizes[0]); idx++) {
        // Seed the random number generator
        srand(time(NULL));

        int N = sizes[idx];

        int **A = allocateMatrix(N);
        int **B = allocateMatrix(N);
        int **result = allocateMatrix(N);

        // Initialize matrices A and B
        initializeMatrix(A, N);
        initializeMatrix(B, N);

        // Start timing
        start = clock();

        // Perform matrix multiplication
        multiplyMatrices(A, B, result, N);

        // End timing
        end = clock();

        // Calculate the elapsed time
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Matrix multiplication for two matrices of size %dx%d took %f seconds.\n", N, N, cpu_time_used);

        // Free allocated memory
        freeMatrix(A, N);
        freeMatrix(B, N);
        freeMatrix(result, N);
    }

    return 0;
}
