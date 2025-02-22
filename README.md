# Web Club Talk  
## Parallel Computing with CUDA
This Repo contains the slides, example code, and references for a talk I held for Web Club NITK members on "Introduction to Parallel Programming using CUDA".  

## Repo Structure
The Repo Structure is as follows:  

```
.
├── CUDA
│   ├── Benchmarks.md // Benchmarks of program speedup
│   ├── cudabook.pdf // A reference book on CUDA  
│   ├── Examples // A folder containing the Example code from the talk  
│   │   ├── ArrayAdd.c
│   │   ├── ArrayAdd.cu
│   │   ├── ArrayAddFaster.cu
│   │   ├── ArrayAddSweep.c 
│   │   ├── MatrixMul.c
│   │   ├── MatrixMul.cu
│   │   ├── MatrixMulSweep.c
│   │   ├── ParallelReduction.cu
│   │   └── Reduction.cpp
│   └── Slides // The slides from the talk
│       ├── Introduction to Parallel Programming using CUDA.pdf
│       └── Introduction to Parallel Programming using CUDA.pptx
└── README.md
```

### Getting Started

1. Clone this repository:

```
git clone https://github.com/yourusername/web-club-cuda-talk.git
```

2. Navigate to the `CUDA/Examples/` directory to explore the code samples.
3. Review the slides in the `CUDA/Slides/` directory for an overview of CUDA concepts.

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- C/C++ compiler compatible with your CUDA version


### Running the Examples

To run the examples after you have installed the CUDA toolkit, run the following commands:
```
nvcc ./CUDA/Examples/[Filename] -o "Nameofoutfile" && ./[Nameofoutfile].out
```

### Additional Resources

- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide For Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [CUDA Installation Guide For Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)



Thank you to Web Club NITK for the opportunity to present this talk :))