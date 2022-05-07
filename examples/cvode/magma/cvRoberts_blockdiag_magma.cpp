/* -----------------------------------------------------------------------------
 * Programmer(s): Cody J. Balos @ LLNL
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
 * The program takes one optional argument, the number of groups
 * of independent ODE systems:
 *
 *    ./cvRoberts_blockdiag_magma [number of groups]
 *
 * This problem is comparable to the cvRoberts_block_klu.c example.
 * --------------------------------------------------------------------------*/

#include <iostream>

#include <cvode/cvode.h>
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>

#if defined(SUNDIALS_MAGMA_BACKENDS_HIP)
#define HIP_OR_CUDA(a,b) a
#elif defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
#define HIP_OR_CUDA(a,b) b
#else
#error "Unsupported MAGMA backend"
#endif

#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
// CUDA vector and memory helper
#include <nvector/nvector_cuda.h>
#include <sunmemory/sunmemory_cuda.h>
// Aliases for CUDA types and constants
constexpr auto gpuDeviceSynchronize = cudaDeviceSynchronize;
constexpr auto gpuGetLastError      = cudaGetLastError;
constexpr auto gpuGetErrorName      = cudaGetErrorName;
const auto gpuSuccess               = cudaSuccess;
const unsigned gpuBlockSize         = 32;
#elif defined(SUNDIALS_MAGMA_BACKENDS_HIP)
// HIP vector and memory helper
#include <nvector/nvector_hip.h>
#include <sunmemory/sunmemory_hip.h>
// Aliases for HIP types and constants
constexpr auto gpuDeviceSynchronize = hipDeviceSynchronize;
constexpr auto gpuGetLastError      = hipGetLastError;
constexpr auto gpuGetErrorName      = hipGetErrorName;
const auto gpuSuccess               = hipSuccess;
const unsigned gpuBlockSize         = 64;
#else
#error "Unsupported MAGMA backend"
#endif

// Problem Constants

#define GROUPSIZE 3            /* number of equations per group */
#define Y1    SUN_RCONST(1.0)      /* initial y components */
#define Y2    SUN_RCONST(0.0)
#define Y3    SUN_RCONST(0.0)
#define RTOL  SUN_RCONST(1.0e-4)   /* scalar relative tolerance            */
#define ATOL1 SUN_RCONST(1.0e-8)   /* vector absolute tolerance components */
#define ATOL2 SUN_RCONST(1.0e-14)
#define ATOL3 SUN_RCONST(1.0e-6)
#define T0    SUN_RCONST(0.0)      /* initial time           */
#define T1    SUN_RCONST(0.4)      /* first output time      */
#define TMULT SUN_RCONST(10.0)     /* output time factor     */
#define NOUT  12               /* number of output times */

#define ZERO  SUN_RCONST(0.0)

// Functions called by the integrator

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

__global__
static void f_kernel(realtype t, realtype* y, realtype* ydot,
                     int neq, int ngroups);

static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

__global__
static void j_kernel(int ngroups, realtype* ydata, realtype *Jdata);

// Functions to check return pointers and values

int check_ptr(void* ptr, const char *funcname);
int check_retval(int value, const char *funcname);

// User-defined data structure

typedef struct {
  int ngroups;
} UserData;


// ------------
// Main Program
// ------------


int main(int argc, char *argv[])
{
  // Create the SUNDIALS context
  sundials::Context sunctx;

  // Default parameter values
  int ngroups = 100;

  // Parse command line arguments
  if (argc > 1) ngroups = atoi(argv[1]);

  // Number of equations
  int neq = ngroups * GROUPSIZE;

  // ---------------------
  // Create the integrator
  // ---------------------

  SUNMemoryHelper memhelper =
    HIP_OR_CUDA( SUNMemoryHelper_Hip(sunctx), SUNMemoryHelper_Cuda(sunctx) );

  // Create initial condition vector
  N_Vector y = HIP_OR_CUDA( N_VNew_Hip(neq, sunctx), N_VNew_Cuda(neq, sunctx) );
  if (check_ptr(y, "N_VNew")) return 1;

  sunrealtype* ydata = N_VGetArrayPointer(y);
  for (int groupj = 0; groupj < neq; groupj += GROUPSIZE)
  {
    ydata[groupj]   = Y1;
    ydata[groupj+1] = Y2;
    ydata[groupj+2] = Y3;
  }
  HIP_OR_CUDA( N_VCopyToDevice_Hip(y), N_VCopyToDevice_Cuda(y) );

  // Create vector of absolute tolerances
  N_Vector abstol = N_VClone(y);
  if (check_ptr(abstol, "N_VClone", 0)) return 1;

  sunrealtype* abstol_data = N_VGetArrayPointer(abstol);

  for (int groupj = 0; groupj < neq; groupj += GROUPSIZE)
  {
    abstol_data[groupj]   = ATOL1;
    abstol_data[groupj+1] = ATOL2;
    abstol_data[groupj+2] = ATOL3;
  }
  HIP_OR_CUDA( N_VCopyToDevice_Hip(abstol), N_VCopyToDevice_Cuda(abstol) );

  // Create and initialize the integrator
  void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (check_ptr(cvode_mem, "CVodeCreate")) return 1;

  int retval = CVodeInit(cvode_mem, f, T0, y);
  if (check_retval(retval, "CVodeInit")) return 1;

  // Create and attach the user data structure
  UserData udata;
  udata.ngroups = ngroups;

  retval = CVodeSetUserData(cvode_mem, &udata);
  if (check_retval(retval, "CVodeSetUserData")) return 1;

  // Specify the scalar relative tolerance and vector absolute tolerances
  retval = CVodeSVtolerances(cvode_mem, RTOL, abstol);
  if (check_retval(retval, "CVodeSVtolerances")) return 1;

  // Create MAGMA block dense SUNMatrix
  SUNMatrix A = SUNMatrix_MagmaDenseBlock(ngroups, GROUPSIZE, GROUPSIZE,
                                          SUNMEMTYPE_DEVICE, memhelper,
                                          NULL, sunctx);
  if (check_ptr(A, "SUNMatrix_MagmaDenseBlock")) return 1;

  // Create the MAGMA SUNLinearSolver object
  SUNLinearSolver LS = SUNLinSol_MagmaDense(y, A, sunctx);
  if (check_ptr(LS, "SUNLinSol_MagmaDense")) return 1;

  // Attach the matrix and linear solver
  retval = CVodeSetLinearSolver(cvode_mem, LS, A);
  if (check_retval(retval, "CVodeSetLinearSolver")) return 1;

  // Set the Jacobian function
  retval = CVodeSetJacFn(cvode_mem, Jac);
  if (check_retval(retval, "CVodeSetJacFn")) return 1;

  // --------------
  // Evolve in time
  // --------------

  std::cout << "Group of independent 3-species kinetics problems" << std::endl;
  std::cout << "Number of groups = " << ngroups << std::endl;

  sunrealtype tret;
  sunrealtype tout = T1;

  for (int iout = 0; iout < NOUT; iout++)
  {
    // Evolve to output time
    retval = CVode(cvode_mem, tout, y, &tret, CV_NORMAL);
    if (check_retval(retval, "CVode")) break;

    // Copy solution to host and print
    HIP_OR_CUDA( N_VCopyFromDevice_Hip(y), N_VCopyFromDevice_Cuda(y) );
    for (int groupj = 0; groupj < ngroups; groupj += 10)
    {
      std::cout << "group " << groupj << " at t = " << tret
                << "y = " << ydata[GROUPSIZE * groupj]
                << " , " << ydata[1+GROUPSIZE*groupj]
                << " , " << ydata[2+GROUPSIZE*groupj]
                << std::endl;
    }

    // Update output time
    tout *= TMULT;
  }

  // Print final statistics
  retval = CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_retval(retval, "CVodePrintAllStats")) return 1;

  // ---------------
  // Destroy objects
  // ---------------

  N_VDestroy(y);
  N_VDestroy(abstol);
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);

  return 0;
}


// ----------------------------------
// Functions called by the integrator
// ----------------------------------


// ODE RHS function launches a GPU kernel to do the actual computation.
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  UserData*    udata    = (UserData*) user_data;
  sunrealtype* ydata    = N_VGetDeviceArrayPointer(y);
  sunrealtype* ydotdata = N_VGetDeviceArrayPointer(ydot);

  unsigned block_size = HIP_OR_CUDA( 64, 32 );
  unsigned grid_size  = (udata->neq + block_size - 1) / block_size;

  f_kernel<<<grid_size, block_size>>>(t, ydata, ydotdata, udata->ngroups);

  gpuDeviceSynchronize();
  auto err = gpuGetLastError();
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
static void f_kernel(realtype t, realtype* ydata, realtype* ydotdata,
                     int ngroups)
{
  for (int groupj = blockIdx.x * blockDim.x + threadIdx.x;
       groupj < ngroups;
       groupj += blockDim.x * gridDim.x)
  {
    sunrealtype y1 = ydata[groupj];
    sunrealtype y2 = ydata[groupj + 1];
    sunrealtype y3 = ydata[groupj + 2];

    ydotdata[groupj]     = SUN_RCONST(-0.04) * y1 + SUN_RCONST(1.0e4) * y2 * y3;
    ydotdata[groupj + 1] = (SUN_RCONST(0.04) * y1 - SUN_RCONST(1.0e4) * y2 * y3
                            - SUN_RCONST(3.0e7) * y2 * y2);
    ydotdata[groupj + 2] = SUN_RCONST(3.0e7) * y2 * y2;
  }
}


// Jacobian function launches a GPU kernel to do the actual computation.
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  UserData*    udata = (UserData*) user_data;
  sunrealtype* Jdata = SUNMatrix_MagmaDense_Data(J);
  sunrealtype* ydata = N_VGetDeviceArrayPointer(y);

  unsigned block_size = HIP_OR_CUDA( 64, 32 );
  unsigned grid_size = (udata->neq + block_size - 1) / block_size;

  j_kernel<<<grid_size, block_size>>>(udata->ngroups, ydata, Jdata);

  gpuDeviceSynchronize();
  auto err = gpuGetLastError();
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
static void j_kernel(int ngroups, realtype* ydata, realtype *Jdata)
{
  int N  = GROUPSIZE;
  int NN = N * N;

  for (int groupj = blockIdx.x * blockDim.x + threadIdx.x;
       groupj < ngroups;
       groupj += blockDim.x * gridDim.x)
  {
    /* get y values */
    sunrealtype y2 = ydata[N * groupj + 1];
    sunrealtype y3 = ydata[N * groupj + 2];

    /* first col of block */
    Jdata[NN * groupj]     = SUN_RCONST(-0.04);
    Jdata[NN * groupj + 1] = SUN_RCONST(0.04);
    Jdata[NN * groupj + 2] = ZERO;

    /* second col of block */
    Jdata[NN * groupj + 3] = SUN_RCONST(1.0e4)  * y3;
    Jdata[NN * groupj + 4] = SUN_RCONST(-1.0e4) * y3 - SUN_RCONST(6.0e7) * y2;
    Jdata[NN * groupj + 5] = SUN_RCONST(6.0e7)  * y2;

    /* third col of block */
    Jdata[NN * groupj + 6] = SUN_RCONST(1.0e4)  * y2;
    Jdata[NN * groupj + 7] = SUN_RCONST(-1.0e4) * y2;
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
