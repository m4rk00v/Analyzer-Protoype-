# GPU Kernel Analyzer – Automated CUDA Kernel Benchmark Orchestrator

The **GPU Kernel Analyzer** is a SLURM-compatible orchestration framework designed to:

1. **Detect CUDA kernels** (`__global__` functions) inside `.cu` files.  
2. **Execute** their corresponding benchmark scripts (`.sh`) for multiple numeric data types (`double`, `float`, `half`).  
3. **Collect performance metrics** such as GFLOPS, memory bandwidth, and execution time.  
4. **Summarize the best-performing configurations** for each kernel in a global report.

It produces both **individual logs per run** and a **global consolidated report** that lists all detected kernels with their optimal block configurations.

---

## Setup Overview

Each CUDA script must reside in its own folder along with its corresponding execution script (`.sh`).  
The main `analyzer.sh` script orchestrates all of them, handling compilation, execution, and log collection automatically.

---

## Step 1 — Organize Scripts and Define Paths

Each CUDA source file should have its own folder containing both the `.cu` and its `.sh` job script.

Example folder structure:

```
/home/jhon/algorithms/
│
├── Poisson/
│   ├── poisson.cu
│   ├── poisson.sh
│
└── Laplace-Algorithm/
    ├── laplace2d_2.cu
    ├── laplace.sh
```

Then, define these paths inside the analyzer:

```bash
ROUTES=(
  "/home/jhon/algorithms/Laplace-Algorithm/laplace2d_2.cu"
  "/home/jhon/algorithms/Poisson/poisson.cu"
)

declare -A JOB_MAP
JOB_MAP["laplace2d_2.cu"]="/home/kevin/algorithms/Laplace-Algorithm/laplace.sh"
JOB_MAP["poisson.cu"]="/home/kevin/algorithms/Poisson/poisson.sh"
```

---

## Step 2 — Add a Kernel Identifier in Each CUDA Source

Every CUDA file must include a **kernel tag** before each `__global__` function you want to benchmark.  
This allows the analyzer to detect kernels automatically.

At the top of each `.cu` file, define:

```cpp
#define KERNEL_TAG /* #[kernel] */
```

Then, mark each kernel like this:

```cpp
#define KERNEL_TAG /* #[kernel] -> top in the main .cu */

KERNEL_TAG
__global__ void jacobi_step(int imax, int jmax,
                            const REAL_T* __restrict__ A,
                            REAL_T* __restrict__ Anew,
                            ERR_T*  __restrict__ err)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > imax || j > jmax) return;

    const int pitch = imax + 2;
    const int id = j * pitch + i;

#if defined(HAS_HALF)
    float Ai = __half2float(A[id]);
    float sum =
        __half2float(A[id+1]) + __half2float(A[id-1]) +
        __half2float(A[id-pitch]) + __half2float(A[id+pitch]);
    float newv_f = 0.25f * sum;
    Anew[id] = __float2half(newv_f);
    err[id]  = fabsf(newv_f - Ai);
#else
    REAL_T Ai   = __ldg(&A[id]);
    REAL_T newv = static_cast<REAL_T>(0.25) * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
                       + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
    Anew[id] = newv;
    err[id]  = real_abs( real_sub(newv, Ai) );
#endif
}
```

---

## Step 3 — Running the Analyzer

Once your folders and tags are set up, launch the analyzer with SLURM:

```bash
sbatch analyzer.sh
```

This single command will:
- Detect all kernels from the specified `.cu` files.
- Launch their associated `.sh` job scripts sequentially.
- Run each job three times: once for `double`, once for `float`, and once for `half` precision.
- Wait for completion and automatically parse the generated logs.

---

## Output Structure

After execution, a new folder is created automatically with a timestamp, containing:

```
kernel_scan_YYYYMMDD-HHMMSS/
│
├── kernel_summary.log      # Human-readable summary
├── kernel_summary.csv      # CSV with all detected kernels
├── kernel_best.csv         # Extracted GLOBAL_BEST lines
└── runs/                   # Raw per-type log files (.out)
```

---

## Example Output

Below is an example of the final report that the analyzer generates in `kernel_summary.log`:

```
==================== FINAL SUMMARY ====================
Total scripts analyzed : 2
Total kernels detected : 3
-------------------------------------------------------
[script] laplace2d_2.cu
    kernels: 1
        - jacobi_step(int imax)
            → [BEST double] bx=128 by=1 GFLOPS=210.256 BW=1401.710GB/s time=0.479ms
            → [BEST float] bx=256 by=4 GFLOPS=377.012 BW=1256.705GB/s time=0.267ms
            → [BEST half] bx=128 by=2 GFLOPS=517.980 BW=863.300GB/s time=0.195ms

[script] poisson.cu
    kernels: 2
        - abs_diff_kernel(int n)
            → [BEST double] bx=16 by=32 GFLOPS=0.000 BW=991.214GB/s time=0.102ms
            → [BEST float] bx=1 by=1024 GFLOPS=0.000 BW=787.219GB/s time=0.064ms
            → [BEST half] bx=1 by=1024 GFLOPS=0.000 BW=739.588GB/s time=0.034ms

        - jacobi_step(int nx)
            → [BEST double] bx=128 by=1 GFLOPS=242.739 BW=2265.565GB/s time=0.103ms
            → [BEST float] bx=256 by=4 GFLOPS=428.895 BW=2001.510GB/s time=0.059ms
            → [BEST half] bx=64 by=8 GFLOPS=593.181 BW=1384.088GB/s time=0.042ms
-------------------------------------------------------
Detailed logs stored in: /home/kevin/algorithms/kernel_scan_20251029-184419/runs
=======================================================
```

---

##  How It Works

- The analyzer **parses each `.cu` file** to find kernel definitions using the `KERNEL_TAG`.
- It then **executes the corresponding `.sh`** file three times for the three numeric types.
- Each job produces `.out` files that contain detailed tuning logs.
- The analyzer extracts every `GLOBAL_BEST` section to generate:
  - `kernel_best.csv` — machine-readable performance data.
  - `kernel_summary.log` — human-readable consolidated report.

---

##  Notes and Tips

- You can safely run `analyzer.sh` on a CPU node; the script handles GPU execution via the sub-job scripts.
- Avoid manually deleting `.out` logs between runs; they are automatically stored under `/runs/`.
- Each `.sh` should compile its `.cu` dynamically using `$TYPE` from environment variables.
- To re-run only parsing (without executing jobs again), set:
  ```bash
  export RUN_JOBS=0
  sbatch analyzer.sh
  ```

---

##  Example Job Script (`poisson.sh`)

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=poisson_job
#SBATCH --output=poisson_out_gpu_cu.txt
#SBATCH --gres=gpu:v100:2

module load nvhpc
TYPE="${TYPE:-double}"

REAL_T_DEF="double"
NVCC_EXTRA=""
case "$TYPE" in
  float)  REAL_T_DEF="float" ;;
  double) REAL_T_DEF="double" ;;
  half)   REAL_T_DEF="__half"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
esac

SRC=poisson.cu
BIN=poisson
ARCH=sm_70

echo "Compiling $SRC -> $BIN (TYPE=$TYPE)"
nvcc -O3 -arch=$ARCH -Xptxas=-v $NVCC_EXTRA -DREAL_T=$REAL_T_DEF "$SRC" -o "$BIN" || exit 2

BX_LIST="1,2,4,8,16,32,64,128,256,512,1024"
BY_LIST="1,2,4,8,16,32,64,128,256,512,1024"

TUNE_REPS=${TUNE_REPS:-2}
WARMUP_ITERS=${WARMUP_ITERS:-0}
PROBE=${PROBE:-1}
TUNE_TRIALS=${TUNE_TRIALS:-3}
EPS_MS=${EPS_MS:-0.02}
NX=${NX:-2048}
NY=${NY:-2048}
MAX_ITERS=${MAX_ITERS:-8000}
TOL=${TOL:-1e-4}
AUTOTUNE=${AUTOTUNE:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CUDA_DEVICE=${CUDA_DEVICE:-0}

srun ./"$BIN" \
     --bx-list="$BX_LIST" \
     --by-list="$BY_LIST" \
     --tune-reps=$TUNE_REPS \
     --warmup-iters=$WARMUP_ITERS \
     --probe=$PROBE \
     --tune-trials=$TUNE_TRIALS \
     --eps-ms=$EPS_MS \
     --device=$CUDA_DEVICE \
     $NX $NY $MAX_ITERS $TOL
```
