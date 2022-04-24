/* -----------------------------------------------------------------------------
 * Programmer(s): Cody J. Balos and David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2022, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * The following is a simple example we simulate a scenario where a set of
 * independent ODEs are combined into batches forming a larger system and groups
 * of these systems are evolved together.
 *
 * For simplicity, each set of ODEs is the same stiff chemical kinetics system,
 * and consists of the following three rate equations:
 *
 *   dy1/dt = -.04 * y1 + 1.0e4 * y2 * y3
 *   dy2/dt =  .04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * (y2)^2
 *   dy3/dt = 3.0e7 * (y2)^2
 *
 * Advanced over the interval from t = 0 to t = 1e11, with initial conditions:
 * y1 = 1.0, y2 = y3 = 0.
 *
 * This program solves the problem with the BDF method, Newton iteration, a
 * user-supplied Jacobian routine, and since the grouping of the independent
 * systems results in a block diagonal linear system, with the MAGMADENSE
 * SUNLinearSolver which supports batched LU factorization. It uses a scalar
 * relative tolerance and a vector absolute tolerance. Output is printed in
 * decades from t = 0.1 to t = 1.0e11. Run statistics (optional outputs) are
 * printed at the end.
 *
 * The program takes two optional argument, the number of batches in a group and
 * the number of groups.
 *
 *   ./cvRoberts_blockdiag_magma [number of batches] [number of groups]
 * --------------------------------------------------------------------------*/

#include <iostream>

#include <cvode/cvode.h>
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>

#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)

// CUDA vector and memory helper
#include <nvector/nvector_cuda.h>
#include <sunmemory/sunmemory_cuda.h>
// Aliases for SUNDIALS and CUDA functions
constexpr auto SUNMemoryHelperNew         = SUNMemoryHelper_Cuda;
constexpr auto NVectorNew                 = N_VNew_Cuda;
constexpr auto NVectorCopyToDevice        = N_VCopyToDevice_Cuda;
constexpr auto NVectorCopyFromDevice      = N_VCopyFromDevice_Cuda;
constexpr auto NVectorSetKernelExecPolicy = N_VSetKernelExecPolicy_Cuda;
constexpr auto SUNGridStrideExecPolicy    = SUNCudaGridStrideExecPolicy;
constexpr auto SUNReduceExecPolicy        = SUNCudaBlockReduceAtomicExecPolicy;
constexpr auto gpuStreamSynchronize       = cudaStreamSynchronize;
constexpr auto gpuGetLastError            = cudaGetLastError;
constexpr auto gpuGetErrorName            = cudaGetErrorName;
constexpr auto gpuStreamCreate            = cudaStreamCreate;
// Aliases for CUDA types and constants
const auto gpuSuccess       = cudaSuccess;
const unsigned gpuBlockSize = 32;
using gpuStream             = cudaStream_t;
using gpuExecPolicy         = SUNCudaExecPolicy;

#elif defined(SUNDIALS_MAGMA_BACKENDS_HIP)

// HIP vector and memory helper
#include <nvector/nvector_hip.h>
#include <sunmemory/sunmemory_hip.h>
// Aliases for SUNDIALS and HIP functions
constexpr auto SUNMemoryHelperNew         = SUNMemoryHelper_Hip;
constexpr auto NVectorNew                 = N_VNew_Hip;
constexpr auto NVectorCopyToDevice        = N_VCopyToDevice_Hip;
constexpr auto NVectorCopyFromDevice      = N_VCopyFromDevice_Hip;
constexpr auto NVectorSetKernelExecPolicy = N_VSetKernelExecPolicy_Hip;
constexpr auto SUNGridStrideExecPolicy    = SUNHipGridStrideExecPolicy;
constexpr auto SUNReduceExecPolicy        = SUNHipBlockReduceExecPolicy;
constexpr auto gpuStreamSynchronize       = hipStreamSynchronize;
constexpr auto gpuGetLastError            = hipGetLastError;
constexpr auto gpuGetErrorName            = hipGetErrorName;
constexpr auto gpuStreamCreate            = hipStreamCreate;
// Aliases for HIP types and constants
const auto gpuSuccess       = hipSuccess;
const unsigned gpuBlockSize = 64;
using gpuStream             = hipStream_t;
using gpuExecPolicy         = SUNHipExecPolicy;

#else
#error "Unsupported MAGMA backend"
#endif

// Functions called by the integrator

int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

__global__
void f_kernel(realtype t, realtype* y, realtype* ydot, int neq_per_group,
              int ngroups);

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

__global__
void j_kernel(int ngroups, realtype* ydata, realtype *Jdata);

// Function to check return pointers and values

int check_ptr(void* ptr, const char *funcname);
int check_retval(int value, const char *funcname);

// User-defined data structure

typedef struct
{
  int batchsize;
  int nbatches;
  int neq_per_group;
  gpuStream stream;
} UserData;


// ------------
// Main Program
// ------------


int main(int argc, char *argv[])
{
  // Default parameter values
  int batchsize = 3;
  int nbatches  = 100;
  int ngroups   = 4;
  int nout      = 12;

  // Parse command line arguments
  if (argc > 1) nbatches = atoi(argv[1]);
  if (argc > 2) ngroups = atoi(argv[1]);

  int neq_per_group = batchsize * nbatches;

  std::cout << "Evolving " << ngroups << " of " << nbatches << " batches of "
            << "independent 3-species kinetics problems" << std::endl;

  int threads_per_block = 1024;
  int blocks_per_grid   = ((neq_per_group + threads_per_block - 1)
                           / threads_per_block);

  // --------------------------
  // Create objects and streams
  // --------------------------

#ifdef _OPENMP
  const int num_threads = omp_get_max_threads();
#else
  const int num_threads = 1;
#endif

  UserData        udata[num_threads];
  SUNContext      sunctx[num_threads];
  SUNMemoryHelper memhelper[num_threads];
  gpuStreams      streams[num_threads];
  N_Vector        y[num_threads];
  N_Vector        abstol[num_threads];
  SUNMatrix       A[num_threads];
  SUNLinearSolver LS[num_threads];
  void*           cvode_mem[num_threads];

  for (int i = 0; i < num_threads; i++)
  {
    // Create user data
    udata[i].batchsize     = batchsize;
    udata[i].nbatches      = nbatches;
    udata[i].neq_per_group = neq_per_group;

    // Create SUNDIALS context
    int retval = SUNContext_Create(nullptr, &sunctx[i]);
    if (check_retval(cvode_mem, "CVodeCreate")) return 1;

    // Create GPU stream
    gpuStreamCreate(&streams[i]);

    // Create memory helper
    memhelper[i] = SUNMemoryHelperNew(sunctx[i]);
    if (check_ptr(memhelper[i], "SUNMemoryHelperNew")) return 1;

    // Create initial condition vector
    y[i] = NVectorNew(neq_per_group, sunctx[i]);
    if (check_ptr(y[i], "NVectorNew")) return 1;

    // Create execution policy
    gpuExecPolicy* stream_exec_policy =
      new SUNGridStrideExecPolicy(threads_per_block, blocks_per_grid,
                                  streams[i]);
    gpuExecPolicy* reduce_exec_policy =
      new SUNReduceExecPolicy(threads_per_block, blocks_per_grid,
                              streams[i]);

    retval = N_VSetKernelExecPolicy_Cuda(y, stream_exec_policy,
                                         reduce_exec_policy);
    if (check_ptr(y[i], "NVectorSetKernelExecPolicy")) return 1;

    delete stream_exec_policy;
    delete reduce_exec_policy;

    // Create absolute tolerance vector
    abstol[i] = N_VClone(y[i]);
    if (check_ptr(abstol[i], "N_VClone")) return 1;

    // Create matrix for use in linear solves
    A[i] = SUNMatrix_MagmaDenseBlock(nbatches, batchsize, batchsize,
                                     SUNMEMTYPE_DEVICE, memhelper[i],
                                     nullptr, sunctx[i]);
    if (check_ptr(A[i], "SUNMatrix_MagmaDenseBlock")) return 1;

    // Create MAGMA linear solver
    LS[i] = SUNLinSol_MagmaDense(y[i], A[i], sunctx[i]);
    if (check_ptr(LS[i], "SUNLinSol_MagmaDense")) return 1;

    // Create and initialize the integrator
    cvode_mem[i] = CVodeCreate(CV_BDF, sunctx[i]);
    if (check_retval(cvode_mem[i], "CVodeCreate")) return 1;

    retval = CVodeInit(cvode_mem[i], f, T0, y[i]);
    if (check_retval(retval, "CVodeInit")) return 1;

    // Attach the user data structure
    retval = CVodeSetUserData(cvode_mem[i], &udata[i]);
    if (check_retval(retval, "CVodeSetUserData")) return 1;

    // Specify the scalar relative tolerance and vector absolute tolerances
    retval = CVodeSVtolerances(cvode_mem[i], RCONST(1.0e-4), abstol[i]);
    if (check_retval(retval, "CVodeSVtolerances")) return 1;

    // Attach the matrix and linear solver
    retval = CVodeSetLinearSolver(cvode_mem[i], LS[i], A[i]);
    if (check_retval(retval, "CVodeSetLinearSolver")) return 1;

    // Set the Jacobian function
    retval = CVodeSetJacFn(cvode_mem[i], Jac);
    if (check_retval(retval, "CVodeSetJacFn")) return 1;
  }

  // --------------
  // Evolve in time
  // --------------

#pragma omp parallel for
  for (int i = 0; i < ngroups; i++)
  {
    // Round-robin across threads/streams
    int idx = i % num_threads;

    // Set the initial condition
    realtype* ydata = N_VGetArrayPointer(y[idx]);
    if (check_ptr(ydata, "N_VGetArrayPointer")) return 1;

    for (int j = 0; j < nbatches; j += batchsize)
    {
      ydata[j]     = RCONST(1.0);
      ydata[j + 1] = RCONST(0.0);
      ydata[j + 2] = RCONST(1.0);
    }
    NVectorCopyToDevice(y[idx]);

    // Set the vector of absolute tolerance
    realtype* abstol_data = N_VGetArrayPointer(abstol[idx]);
    if (check_ptr(abstol_data, "N_VGetArrayPointer")) return 1;

    for (int j = 0; j < nbatches; j += batchsize)
    {
      abstol_data[j]     = RCONST(1.0e-8);
      abstol_data[j + 1] = RCONST(1.0e-14);
      abstol_data[j + 2] = RCONST(1.0e-6);
    }
    NVectorCopyToDevice(abstol[idx]);

    realtype tret;
    realtype tout = RCONST(0.1);

    for (int iout = 0; iout < nout; iout++)
    {
      // Advance toward tout
      int retval = CVode(cvode_mem[idx], tout, y[idx], &tret, CV_NORMAL);
      if (check_retval(retval, "CVode")) return 1;

      // Update output time
      tout *= RCONST(10.0);
    }

    // Print final statistics
#pragma omp critical
    {
      retval = CVodePrintAllStats(cvode_mem[idx], stdout,
                                  SUN_OUTPUTFORMAT_TABLE);
      if (check_retval(retval, "CVodePrintAllStats")) return 1;
    }
  }

  // ---------------------------
  // Destroy objects and streams
  // ---------------------------

  for (int i = 0; i < num_threads; i++)
  {
    N_VDestroy(y[i]);
    N_VDestroy(abstol[i]);
    SUNMatDestroy(A[i]);
    SUNLinSolFree(LS[i]);
    CVodeFree(&cvode_mem[i]);
    cudaStreamDestroy(streams[i]);
  }

  return 0;
}


// ----------------------------------
// Functions called by the integrator
// ----------------------------------


// ODE RHS function launches a GPU kernel to do the actual computation.
int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  UserData* udata    = (UserData*) user_data;
  realtype* ydata    = N_VGetDeviceArrayPointer(y);
  realtype* ydotdata = N_VGetDeviceArrayPointer(ydot);

  unsigned block_size = gpuBlockSize;
  unsigned grid_size  = (udata->neq_per_group + block_size - 1) / block_size;

  f_kernel<<<grid_size, block_size>>>(t, ydata, ydotdata, udata->neq_per_group,
                                      udata->batchsize);

  gpuStreamSynchronize(udata->stream);
  auto err = gpuGetLastError();,

  if (err != gpuSuccess)
  {
    std::cerr << ">>> ERROR in f: gpuGetLastError returned "
              << gpuGetErrorName(err) << std::endl;
    return -1;
  }

  return 0;
}


// Right hand side function evalutation kernel
__global__
void f_kernel(realtype t, realtype* ydata, realtype* ydotdata,
              int neq_per_group, int batchsize)
{
  realtype y1, y2, y3, yd1, yd3;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int groupj = i * batchsize;

  if (i < neq_per_group)
  {
    y1 = ydata[groupj];
    y2 = ydata[groupj + 1];
    y3 = ydata[groupj + 2];

    yd1 = ydotdata[groupj]   = RCONST(-0.04) * y1 + RCONST(1.0e4) * y2 * y3;
    yd3 = ydotdata[groupj+2] = RCONST(3.0e7) * y2 * y2;
    ydotdata[groupj+1] = -yd1 - yd3;
  }
}


// Jacobian function launches a GPU kernel to do the actual computation.
int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  UserData* udata = (UserData*) user_data;
  realtype* Jdata = SUNMatrix_MagmaDense_Data(J);
  realtype* ydata = N_VGetDeviceArrayPointer(y);

  unsigned block_size = gpuBlockSize;
  unsigned grid_size = (udata->neq_per_group + block_size - 1) / block_size;

  J_kernel<<<grid_size, block_size>>>(udata->ngroups, ydata, Jdata);

  gpuStreamSynchronize(udata->stream);
  auto err = gpuGetLastError();,

  if (err != gpuSuccess)
  {
    std::cerr << ">>> ERROR in f: gpuGetLastError returned "
              << gpuGetErrorName(err) << std::endl;
    return -1;
  }

  return 0;
}


// Jacobian evaluation GPU kernel
__global__
void J_kernel(int ngroups, realtype* ydata, realtype *Jdata)
{
  int N  = GROUPSIZE;
  int NN = N * N;
  int groupj;
  realtype y2, y3;

  for (groupj = blockIdx.x * blockDim.x + threadIdx.x;
       groupj < ngroups;
       groupj += blockDim.x * gridDim.x)
  {
    /* get y values */
    y2 = ydata[N * groupj + 1];
    y3 = ydata[N * groupj + 2];

    /* first col of block */
    Jdata[NN * groupj]     = RCONST(-0.04);
    Jdata[NN * groupj + 1] = RCONST(0.04);
    Jdata[NN * groupj + 2] = ZERO;

    /* second col of block */
    Jdata[NN * groupj + 3] = RCONST(1.0e4) * y3;
    Jdata[NN * groupj + 4] = (RCONST(-1.0e4) * y3) - (RCONST(6.0e7) * y2);
    Jdata[NN * groupj + 5] = RCONST(6.0e7) * y2;

    /* third col of block */
    Jdata[NN * groupj + 6] = RCONST(1.0e4) * y2;
    Jdata[NN * groupj + 7] = RCONST(-1.0e4) * y2;
    Jdata[NN * groupj + 8] = ZERO;
  }
}


// ----------------
// Helper functions
// ----------------


// Check function return pointer
int check_ptr(void* ptr, const char *funcname)
{
  if (!ptr)
  {
    std::cerr << "ERROR: " << funcname << " returned a NULL pointer"
              << std::endl;
    return 1;
  }
  return 0;
}


// Check function return value
int check_retval(int value, const char *funcname)
{
  if (value < 0)
  {
    std::cerr << "ERROR: " << funcname << " returned " << value << std::endl;
    return 1;
  }
  return 0;
}
