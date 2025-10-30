# GPU Kernel Analyzer – Automated CUDA Kernel Benchmark Orchestrator

The **Kernel Analyzer** is a SLURM-compatible orchestration script designed to automatically:

1. Detect CUDA kernels (`__global__` functions) inside target `.cu` files.  
2. Launch their corresponding job scripts (`.sh`) for multiple data types (`double`, `float`, `half`).  
3. Collect performance metrics (GFLOPS, Bandwidth, Runtime) for each kernel and report the **best-performing configuration** per data type.

The analyzer produces both **individual logs** for each run and a **global summary report** showing all detected kernels and their best configurations.

---

## 🧩 Setup Instructions

### 1. Each Scripts should be in its carpet with its .sh 

### 2. Define the CUDA scripts and job mappings

Edit the `ROUTES` and `JOB_MAP` sections at the top of `analyzer.sh`:

```bash
ROUTES=(
  "/home/kevin/algorithms/Laplace-Algorithm/laplace2d_2.cu"
  "/home/kevin/algorithms/Poisson/poisson.cu"
)

declare -A JOB_MAP
JOB_MAP["laplace2d_2.cu"]="/home/kevin/algorithms/Laplace-Algorithm/laplace.sh"
JOB_MAP["poisson.cu"]="/home/kevin/algorithms/Poisson/poisson.sh"


# Kernel Tag Convention

Each CUDA source file (`.cu`) that contains GPU kernels must include a **kernel identifier** so that the analyzer can automatically detect and register all kernels.  
This identifier works as a decorator tag placed before each kernel function.

---

## Define the Kernel Tag

At the top of your CUDA source file, include the following line:

```cpp
#define KERNEL_TAG /* #[kernel] */


example:

#define KERNEL_TAG /* #[kernel] -> top in the main .cu */

KERNEL_TAG
__global__ void jacobi_step(int imax, int jmax,
                            const REAL_T* __restrict__ A,
                            REAL_T* __restrict__ Anew,
                            ERR_T*  __restrict__ err)
{
