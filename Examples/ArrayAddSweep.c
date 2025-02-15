#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_VALUE 10 // Maximum value for random numbers

int main() {
    int sizes[10] = {1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000};
    int i;

    for(int idx = 0; idx<(sizeof(sizes)/sizeof(sizes[0])); idx++) {

        int N = sizes[idx];
        int *arr1, *arr2, *result;

        clock_t start, end;
        double cpu_time_used;

        // Seed the random number generator
        srand(time(NULL));

        // Dynamically allocate memory for the arrays
        arr1 = (int*)malloc(N * sizeof(int));
        arr2 = (int*)malloc(N * sizeof(int));
        result = (int*)malloc(N * sizeof(int));

        if (arr1 == NULL || arr2 == NULL || result == NULL) {
            printf("Memory allocation failed\n");
            return 1;
        }

        // Generate random numbers for the arrays
        for (i = 0; i < N; i++) {
            arr1[i] = rand() % MAX_VALUE;
            arr2[i] = rand() % MAX_VALUE;
        }

        // Start timing
        start = clock();

        // Add the arrays
        for (i = 0; i < N; i++) {
            result[i] = arr1[i] + arr2[i];
        }

        // Free dynamically allocated memory
        free(arr1);
        free(arr2);
        free(result);

        // End timing
        end = clock();

        // Calculate the elapsed time
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        
        printf("Array addition of 2 arrays of size %d took %f seconds.\n",N,cpu_time_used);
    
    }

    return 0;
}
