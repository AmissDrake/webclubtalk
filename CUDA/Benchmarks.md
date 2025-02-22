# Benchmarks

## Array Addition

### Serial
```
Array addition of 2 arrays of size 1 took 0.000001 seconds.  
Array addition of 2 arrays of size 10 took 0.000001 seconds.  
Array addition of 2 arrays of size 100 took 0.000001 seconds.  
Array addition of 2 arrays of size 1000 took 0.000003 seconds.  
Array addition of 2 arrays of size 10000 took 0.000028 seconds.  
Array addition of 2 arrays of size 100000 took 0.000672 seconds.  
Array addition of 2 arrays of size 1000000 took 0.005071 seconds.  
Array addition of 2 arrays of size 10000000 took 0.050351 seconds.  
Array addition of 2 arrays of size 100000000 took 0.393491 seconds.  
Array addition of 2 arrays of size 1000000000 killed my pc xD. (Not enough RAM)
```

### Parallel
```
The program to add 2 arrays of size 1000000000 took 0.169441 seconds.  
The program to add 2 arrays of size 1000000000 actually took 0.030317 seconds.
```  
## Matrix Multiplication

### Serial
```
Matrix multiplication for two matrices of size 1x1 took 0.000001 seconds.  
Matrix multiplication for two matrices of size 10x10 took 0.000006 seconds.  
Matrix multiplication for two matrices of size 100x100 took 0.002921 seconds.  
Matrix multiplication for two matrices of size 1000x1000 took 4.750118 seconds.  
Matrix multiplication for two matrices of size 1500x1500 took 15.838012 seconds.  
Matrix multiplication for two matrices of size 1800x1800 took 34.588351 seconds.  
```

### Parallel
```
Matrix multiplication of 2 1800x1800 matrices took 0.019275 seconds.  
Matrix multiplication of 2 1800x1800 matrices took 0.076550 seconds including memory operations.
```

## Parallel Reduction
Reducing an array of 1s of size 1,000,000,000  
```
Sum: 1000000000.000000
 Calculation of the sum took 0.034915 seconds.
 Calculation of the sum took 2.704010 seconds including memory operations.
 ```