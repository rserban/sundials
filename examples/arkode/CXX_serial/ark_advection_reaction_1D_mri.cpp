/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2020, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a brusselator problem from chemical kinetics.
 * This is n PDE system with 3 components, Y = [u,v,w], satisfying the
 * equations,
 *
 *    u_t = -au * u_x +  a - (w + 1) * u + v * u^2
 *    v_t = -av * v_x +  w * u - v * u^2
 *    w_t = -aw * w_x + (b - w) / ep - w * u
 *
 * for t in [0, 10], x in [0, 1], with initial conditions
 *
 *    u(0,x) =  a  + 0.1 * sin(pi * x)
 *    v(0,x) = b/a + 0.1 * sin(pi * x)
 *    w(0,x) =  b  + 0.1 * sin(pi * x),
 *
 * and with stationary boundary conditions, i.e.
 *
 *    u_t(t,0) = u_t(t,1) = 0,
 *    v_t(t,0) = v_t(t,1) = 0,
 *    w_t(t,0) = w_t(t,1) = 0.
 *
 * Note: these can also be implemented as Dirichlet boundary conditions with
 * values identical to the initial conditions.
 *
 * We use parameters:
 *
 *   au = av = aw = 0.001 (advection coefficients - velocity)
 *   a  = 0.6
 *   b  = 2
 *   ep = 0.01
 *
 * The spatial derivatives are computed using second-order
 * centered differences, with the data distributed over N points
 * on a uniform spatial grid.
 * Note: larger values of advection require advection schemes such as
 * upwinding not implemented here.
 *
 * We use Newton iteration with the SUNBAND linear solver and a user supplied
 * Jacobian routine for nonlinear solves.
 *
 * This program solves the problem with the MRI stepper. 10 outputs are printed
 * at equal intervals, and run statistics are printed at the end.
 * ---------------------------------------------------------------------------*/

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <string>
#include <chrono>
#include <cmath>

#include <arkode/arkode_mristep.h>    // prototypes for MRIStep fcts., consts
#include <arkode/arkode_arkstep.h>    // prototypes for ARKStep fcts., consts
#include <nvector/nvector_serial.h>   // access to Serial N_Vector
#include <sunmatrix/sunmatrix_band.h> // access to band SUNMatrix
#include <sunlinsol/sunlinsol_band.h> // access to band SUNLinearSolver
#include <sundials/sundials_types.h>  // def. of type 'realtype'

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif


// Macros for problem constants
#define ZERO  RCONST(0.0)
#define ONE   RCONST(1.0)
#define TWO   RCONST(2.0)
#define PI    RCONST(3.141592653589793238462643383279502884197169)

// Accessor macros between 1D array and species v at location x
#define IDX(x,v) (3*(x)+v)

using namespace std;

// -----------------------------------------------------------------------------
// Utility data structures and classes
// -----------------------------------------------------------------------------

// User data structure and default values
struct UserData
{
  sunindextype N   = 201;             // number of nodes
  sunindextype NEQ = 3 * N;           // number of equations
  realtype     dx  = ONE / (N - 1);   // mesh spacing
  realtype     a   = RCONST(0.6);     // constant forcing on u
  realtype     b   = RCONST(2.0);     // steady-state value of w
  realtype     ep  = RCONST(0.01);    // stiffness parameter
  realtype     au  = RCONST(0.001);   // advection coeff for u
  realtype     av  = RCONST(0.001);   // advection coeff for v
  realtype     aw  = RCONST(0.001);   // advection coeff for w
};


// Problem/integrator options and default values
struct UserOptions
{
  realtype T0     = ZERO;             // initial time
  realtype Tf     = RCONST(3.0);      // final time
  int      method = 3;                // MRI method
  int      order  = 3;                // method order
  int      m      = 5;                // time scale separation
  realtype hs     = RCONST(0.1);      // slow time step
  realtype hf     = hs / m;           // fast time step
  realtype reltol = RCONST(1.0e-8);   // relative tolerance
  realtype abstol = RCONST(1.0e-12);  // absolute tolerance
};

// Output data structure
struct OutputData
{
  int          Nt = 10;              // number of output times
  realtype     dTout;                // Time between outputs
  sunindextype N;                    // number of nodes
  sunindextype NEQ;                  // vector length
  ofstream     yout;                 // solution output stream
  N_Vector     umask, vmask, wmask;  // mask vectors for output
};


// Simple timer class
class Timer
{
public:
  Timer() : total_(0.0) {}
  void start() { start_ = chrono::steady_clock::now(); }
  void stop()
  {
    end_ = chrono::steady_clock::now();
    total_ += chrono::duration<double>(end_ - start_).count();
  }
  double total() const { return total_; }
private:
  double total_;
  chrono::time_point<chrono::steady_clock> start_;
  chrono::time_point<chrono::steady_clock> end_;
};

// -----------------------------------------------------------------------------
// Functions provided to the SUNDIALS integrator
// -----------------------------------------------------------------------------

// ODE Rhs functions
static int RhsAdvection(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int RhsReaction(realtype t, N_Vector y, N_Vector ydot, void *user_data);

// Rhs Jacobian functions
static int JacReaction(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                       void *user_data, N_Vector tmp1, N_Vector tmp2,
                       N_Vector tmp3);

// -----------------------------------------------------------------------------
// Problem evolution functions
// -----------------------------------------------------------------------------

static int EvolveARK(N_Vector y, UserData *udata, UserOptions *uopts,
                     OutputData *outdata);

static int EvolveMRI(N_Vector y, UserData *udata, UserOptions *uopts,
                     OutputData *outdata);

static int EvolveLT(N_Vector y, UserData *udata, UserOptions *uopts,
                    OutputData *outdata);

static int LTStepEvolve(void *arkode_mem, void *inner_arkode_mem, realtype hs,
                        realtype dTout, realtype tout, N_Vector y, realtype *t);

static int EvolveSM(N_Vector y, UserData *udata, UserOptions *uopts,
                    OutputData *outdata);

static int SMStepEvolve(void *arkode_mem, void *inner_arkode_mem, realtype hs,
                        realtype dTout, realtype tout, N_Vector y, realtype *t);

// -----------------------------------------------------------------------------
// Output and utility functions
// -----------------------------------------------------------------------------

// Read the command line inputs
static int ReadInputs(int *argc, char ***argv, UserData *udata,
                      UserOptions *uopts, OutputData *outdata);

// Set the initial condition
static int SetIC(N_Vector y, void *user_data);

// Output solution and error
static int OpenOutput(N_Vector y, OutputData *outdata);
static int WriteOutput(realtype t, N_Vector y, OutputData *outdata);
static int CloseOutput(OutputData *outdata);

// Print integration statistics
static int OutputStatsARK(void *arkode_mem);
static int OutputStatsMRI(void *arkode_mem, void* inner_arkode_mem);
static int OutputStatsLT(void *arkode_mem, void* inner_arkode_mem);
static int OutputStatsSM(void *arkode_mem, void* inner_arkode_mem);

// Check function return values
static int check_retval(void *returnvalue, const char *funcname, int opt);

// -----------------------------------------------------------------------------
// Main Program
// -----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  // Start timer
  Timer overall;
  overall.start();

  // Reusable error flag
  int retval;

  // Create user data, options, and output structures
  UserData    udata;
  UserOptions uopts;
  OutputData  outdata;

  // Parse command line inputs
  retval = ReadInputs(&argc, &argv, &udata, &uopts, &outdata);
  if (check_retval(&retval, "ReadInputs", 1)) return 1;

  // Initial problem output
  cout << endl;
  cout << "1D Advection-Reaction (Brusselator) test problem:" << endl;
  cout << "  time domain: [" << uopts.T0 << "," << uopts.Tf << "]" << endl;
  cout << "  spatial domain: [" << 0 << "," << 1 << "]" << endl;
  cout << "    nodes = " << udata.N << " (NEQ = " << udata.NEQ << ")" << endl;
  cout << "    dx    = " << udata.dx << endl;
  cout << "  problem parameters:" << endl;
  cout << "    a  = " << udata.a  << endl;
  cout << "    b  = " << udata.b  << endl;
  cout << "    ep = " << udata.ep << endl;
  cout << "  advection coefficients:" << endl;
  cout << "    au = " << udata.au << endl;
  cout << "    av = " << udata.av << endl;
  cout << "    aw = " << udata.aw << endl;
  cout << "  integrator settings:" << endl;
  cout << "    hs     = " << uopts.hs << endl;
  cout << "    hf     = " << uopts.hf << endl;
  cout << "    m      = " << uopts.m  << endl;
  cout << "    reltol = " << uopts.reltol << endl;
  cout << "    abstol = " << uopts.abstol << endl;
  cout << endl;

  // Create solution vector
  N_Vector y = N_VNew_Serial(udata.NEQ);
  if (check_retval((void *)y, "N_VNew_Serial", 0)) return 1;

  // Set initial condition
  retval = SetIC(y, &udata);
  if (check_retval(&retval, "SetIC", 1)) return 1;

  // Evolve with the desired method
  switch (uopts.method)
  {
  case(0):
    cout << "Integrating with ARKStep" << endl << endl;
    retval = EvolveARK(y, &udata, &uopts, &outdata);
    if (check_retval(&retval, "EvolvARK", 1)) return 1;
    break;
  case(1):
    cout << "Integrating with Lie-Trotter splitting" << endl << endl;
    retval = EvolveLT(y, &udata, &uopts, &outdata);
    if (check_retval(&retval, "EvolveLT", 1)) return 1;
    break;
  case(2):
    cout << "Integrating with Strang-Marchuk splitting" << endl << endl;
    retval = EvolveSM(y, &udata, &uopts, &outdata);
    if (check_retval(&retval, "EvolveSM", 1)) return 1;
    break;
  case(3):
    cout << "Integrating with MRIStep, ";
    retval = EvolveMRI(y, &udata, &uopts, &outdata);
    if (check_retval(&retval, "EvolveMRI", 1)) return 1;
    break;
  default:
    cerr << "ERROR: invalid method" << endl;
    break;
  }

  // Free solution vector
  N_VDestroy(y);

  // Output total runtime
  overall.stop();
  cout << endl << "Total Runtime: " << overall.total() << endl;

  return 0;
}


// -----------------------------------------------------------------------------
// Evolve with ARK
// -----------------------------------------------------------------------------


static int EvolveARK(N_Vector y, UserData *udata, UserOptions *uopts,
                     OutputData *outdata)
{
  // Reusable error flag
  int retval;

  // Integrator data and settings
  void *arkode_mem = ARKStepCreate(RhsAdvection, RhsReaction, uopts->T0, y);
  if (check_retval((void *) arkode_mem, "ARKStepCreate", 0)) return 1;

  // Set method order to use
  retval = ARKStepSetOrder(arkode_mem, 5);
  if (check_retval(&retval, "ARKStepSetOrder",1)) return 1;

  // Set the step size
  retval = ARKStepSetFixedStep(arkode_mem, uopts->hs);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Specify fast tolerances
  retval = ARKStepSStolerances(arkode_mem, uopts->reltol, uopts->abstol);
  if (check_retval(&retval, "ARKStepSStolerances", 1)) return 1;

  // Initialize matrix and linear solver data structures
  SUNMatrix A = SUNBandMatrix(udata->NEQ, 3, 3);
  if (check_retval((void *)A, "SUNBandMatrix", 0)) return 1;

  SUNLinearSolver LS = SUNLinSol_Band(y, A);
  if (check_retval((void *)LS, "SUNLinSol_Band", 0)) return 1;

  // Attach matrix and linear solver
  retval = ARKStepSetLinearSolver(arkode_mem, LS, A);
  if (check_retval(&retval, "ARKStepSetLinearSolver", 1)) return 1;

  // Set max number of nonlinear iters
  retval = ARKStepSetMaxNonlinIters(arkode_mem, 10);
  if (check_retval(&retval, "ARKStepSetMaxNonlinIters", 1)) return 1;

  // Set the Jacobian routine
  retval = ARKStepSetJacFn(arkode_mem, JacReaction);
  if (check_retval(&retval, "ARKStepSetJacFn", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // Set maximum number of steps taken by solver
  retval = ARKStepSetMaxNumSteps(arkode_mem, 100000000);
  if (check_retval(&retval, "ARKStepSetMaxNumSteps", 1)) return 1;

  // -------------
  // Integrate ODE
  // -------------

  // Open output files
  retval = OpenOutput(y, outdata);
  if (check_retval(&retval, "OpenOutput", 1)) return 1;

  // Set initial time and first output time
  realtype t    = uopts->T0;
  realtype tout = uopts->T0 + outdata->dTout;

  // Output the initial condition
  retval = WriteOutput(t, y, outdata);
  if (check_retval(&retval, "WriteOutput", 1)) return 1;

  // Main time-stepping
  Timer evolve;

  for (int iout = 0; iout < outdata->Nt; iout++)
  {
    // Stop at output time (do not interpolate)
    retval = ARKStepSetStopTime(arkode_mem, tout);
    if (check_retval(&retval, "ARKStepSetStopTime", 1)) return 1;

    // Advance in time
    evolve.start();
    retval = ARKStepEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);
    evolve.stop();
    if (check_retval(&retval, "ARKStepEvolve", 1)) break;

    // Write output
    retval = WriteOutput(t, y, outdata);
    if (check_retval(&retval, "WriteOutput", 1)) break;

    // Update output time
    tout += outdata->dTout;
    tout = (tout > uopts->Tf) ? uopts->Tf : tout;
  }

  // Close output
  retval = CloseOutput(outdata);
  if (check_retval(&retval, "CloseOutput", 1)) return 1;

  // --------
  // Finalize
  // --------

  // Output integration stats
  retval = OutputStatsARK(arkode_mem);
  if (check_retval(&retval, "OutputStats", 1)) return 1;

  cout << "Timing:" << endl;
  cout << "  Evolution: " << evolve.total() << endl;

  // Clean up
  ARKStepFree(&arkode_mem); // Free integrator memory
  SUNMatDestroy(A);         // Free fast matrix
  SUNLinSolFree(LS);        // Free fast linear solver

  return 0;
}


static int OutputStatsARK(void *arkode_mem)
{
  int retval;

  long int nst, nfe, nfi, nni, nnc, nje;

  // Get some fast integrator statistics
  retval = ARKStepGetNumSteps(arkode_mem, &nst);
  if (check_retval(&retval, "ARKStepGetNumSteps", 1)) return 1;

  retval = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  if (check_retval(&retval, "ARKStepGetNumRhsEvals", 1)) return 1;

  // Print some final statistics
  cout << endl;
  cout << "Final Solver Statistics:" << endl;
  cout << "  Steps = " << nst << endl;
  cout << "  Ex Rhs evals = " << nfe << endl;
  cout << "  Im Rhs evals = " << nfi << endl;

  // Get/print fast integrator implicit solver statistics
  retval = ARKStepGetNonlinSolvStats(arkode_mem, &nni, &nnc);
  if (check_retval(&retval, "ARKStepGetNonlinSolvStats", 1)) return 1;

  retval = ARKStepGetNumJacEvals(arkode_mem, &nje);
  if (check_retval(&retval, "ARKStepGetNumJacEvals", 1)) return 1;

  cout << "  Newton iters      = " << nni << endl;
  cout << "  Newton conv fails = " << nnc << endl;
  cout << "  Jacobian evals    = " << nje << endl;
  cout << endl;

  return 0;
}


// -----------------------------------------------------------------------------
// Evolve with MRI
// -----------------------------------------------------------------------------


static int EvolveMRI(N_Vector y, UserData *udata, UserOptions *uopts,
                     OutputData *outdata)
{
  // Reusable error flag
  int retval;

  // -------------------------
  // Setup the fast integrator
  // -------------------------

  // Implicit reactions
  void *inner_arkode_mem = ARKStepCreate(NULL, RhsReaction, uopts->T0, y);
  if (check_retval((void *) inner_arkode_mem, "ARKStepCreate", 0)) return 1;

  // Set fast method table
  ARKodeButcherTable B;
  realtype gamma, beta;

  switch (uopts->order)
  {
  case(2):
    // sdirk-2-2
    cout << "2nd order" << endl << endl;
    B = ARKodeButcherTable_Alloc(2, SUNFALSE);
    if (check_retval((void *)B, "ARKodeButcherTable_Alloc", 0)) return 1;

    gamma = (TWO - sqrt(TWO)) / TWO;

    B->A[0][0] = gamma;
    B->A[1][0] = ONE - gamma;
    B->A[1][1] = gamma;
    B->b[0]    = ONE - gamma;
    B->b[1]    = gamma;
    B->c[0]    = gamma;
    B->c[1]    = ONE;
    B->q       = 2;
    break;
  case(3):
    // esdirk-3-3
    cout << "3rd order" << endl << endl;
    B = ARKodeButcherTable_Alloc(3, SUNFALSE);
    if (check_retval((void *)B, "ARKodeButcherTable_Alloc", 0)) return 1;

    beta  = sqrt(RCONST(3.0)) / RCONST(6.0) + RCONST(0.5);
    gamma = (-ONE/RCONST(8.0)) * (sqrt(RCONST(3.0)) + ONE);

    B->A[1][0] = RCONST(4.0) * gamma + TWO * beta;
    B->A[1][1] = ONE - RCONST(4.0) * gamma - TWO * beta;
    B->A[2][0] = RCONST(0.5) - beta - gamma;
    B->A[2][1] = gamma;
    B->A[2][2] = beta;
    B->b[0]    = ONE / RCONST(6.0);
    B->b[1]    = ONE / RCONST(6.0);
    B->b[2]    = TWO / RCONST(3.0);
    B->c[1]    = ONE;
    B->c[2]    = RCONST(0.5);
    B->q       = 3;
    break;
  case(4):
    cout << "4th order" << endl << endl;
    B = ARKodeButcherTable_LoadDIRK(CASH_5_3_4);
    if (check_retval((void *)B, "ARKodeButcherTable_LoadDIRK", 0)) return 1;
    break;
  default:
    cerr << "ERROR: invalid method order" << endl;
    break;
  }

  retval = ARKStepSetTables(inner_arkode_mem, uopts->order, 0, B, NULL);
  if (check_retval(&retval, "ARKStepSetTables", 1)) return 1;

  // Set the fast step size
  retval = ARKStepSetFixedStep(inner_arkode_mem, uopts->hf);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Specify fast tolerances
  retval = ARKStepSStolerances(inner_arkode_mem, uopts->reltol, uopts->abstol);
  if (check_retval(&retval, "ARKStepSStolerances", 1)) return 1;

  // Initialize matrix and linear solver data structures
  SUNMatrix A = SUNBandMatrix(udata->NEQ, 3, 3);
  if (check_retval((void *)A, "SUNBandMatrix", 0)) return 1;

  SUNLinearSolver LS = SUNLinSol_Band(y, A);
  if (check_retval((void *)LS, "SUNLinSol_Band", 0)) return 1;

  // Attach matrix and linear solver
  retval = ARKStepSetLinearSolver(inner_arkode_mem, LS, A);
  if (check_retval(&retval, "ARKStepSetLinearSolver", 1)) return 1;

  // Set the Jacobian routine
  retval = ARKStepSetJacFn(inner_arkode_mem, JacReaction);
  if (check_retval(&retval, "ARKStepSetJacFn", 1)) return 1;

  // Set max number of nonlinear iters
  retval = ARKStepSetMaxNonlinIters(inner_arkode_mem, 10);
  if (check_retval(&retval, "ARKStepSetMaxNonlinIters", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(inner_arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // -------------------------
  // Setup the slow integrator
  // -------------------------

  // Explicit slow (default MIS method)
  void *arkode_mem = MRIStepCreate(RhsAdvection, uopts->T0, y, MRISTEP_ARKSTEP,
                                   inner_arkode_mem);
  if (check_retval((void *)arkode_mem, "MRIStepCreate", 0)) return 1;

  // Set slow method table
  MRIStepCoupling C;
  ARKodeButcherTable Bc;

  switch (uopts->order)
  {
  case(2):
    // MRI-GARK-ERK22
    Bc = ARKodeButcherTable_Alloc(2, SUNFALSE);
    if (check_retval((void *)Bc, "MRIStepCoupling_Alloc", 0)) return 1;
    Bc->A[1][0] = RCONST(0.5);
    Bc->b[1] = ONE;
    Bc->c[1] = RCONST(0.5);
    Bc->q = 2;
    C = MRIStepCoupling_MIStoMRI(Bc, 2, 0);
    if (check_retval((void *)C, "MRIStepCoupling_MIStoMRI", 0)) return 1;
    ARKodeButcherTable_Free(Bc);
    break;
  case(3):
    C = MRIStepCoupling_LoadTable(MIS_KW3);
    if (check_retval((void *)C, "MRIStepCoupling_LoadTable", 0)) return 1;
    break;
  case(4):
    C = MRIStepCoupling_LoadTable(MRI_GARK_ERK45a);
    if (check_retval((void *)C, "MRIStepCoupling_LoadTable", 0)) return 1;
    break;
  default:
    cerr << "ERROR: invalid method order" << endl;
    break;
  }

  retval = MRIStepSetCoupling(arkode_mem, C);
  if (check_retval(&retval, "MRIStepSetCoupling", 1)) return 1;

  // Set the slow step size
  retval = MRIStepSetFixedStep(arkode_mem, uopts->hs);
  if (check_retval(&retval, "MRIStepSetFixedStep", 1)) return 1;

  // Set maximum number of steps taken by solver
  retval = MRIStepSetMaxNumSteps(arkode_mem, 100000000);
  if (check_retval(&retval, "MRIStepSetMaxNumSteps", 1)) return 1;

  // Pass udata to user functions
  retval = MRIStepSetUserData(arkode_mem, (void *) udata);
  if (check_retval(&retval, "MRIStepSetUserData", 1)) return 1;

  // -------------
  // Integrate ODE
  // -------------

  // Open output files
  retval = OpenOutput(y, outdata);
  if (check_retval(&retval, "OpenOutput", 1)) return 1;

  // Set initial time and first output time
  realtype t    = uopts->T0;
  realtype tout = uopts->T0 + outdata->dTout;

  // Output the initial condition
  retval = WriteOutput(t, y, outdata);
  if (check_retval(&retval, "WriteOutput", 1)) return 1;

  // Main time-stepping
  Timer evolve;

  for (int iout = 0; iout < outdata->Nt; iout++)
  {
    // Stop at output time (do not interpolate)
    retval = MRIStepSetStopTime(arkode_mem, tout);
    if (check_retval(&retval, "MRIStepSetStopTime", 1)) return 1;

    // Advance in time
    evolve.start();
    retval = MRIStepEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);
    evolve.stop();
    if (check_retval(&retval, "MRIStepEvolve", 1)) break;

    // Write output
    retval = WriteOutput(t, y, outdata);
    if (check_retval(&retval, "WriteOutput", 1)) break;

    // Update output time
    tout += outdata->dTout;
    tout = (tout > uopts->Tf) ? uopts->Tf : tout;
  }

  // Close output
  retval = CloseOutput(outdata);
  if (check_retval(&retval, "CloseOutput", 1)) return 1;

  // --------
  // Finalize
  // --------

  // Output integration stats
  retval = OutputStatsMRI(arkode_mem, inner_arkode_mem);
  if (check_retval(&retval, "OutputStats", 1)) return 1;

  cout << "Timing:" << endl;
  cout << "  Evolution: " << evolve.total() << endl;

  // Clean up
  ARKStepFree(&inner_arkode_mem);   // Free integrator memory
  MRIStepFree(&arkode_mem);         // Free integrator memory
  ARKodeButcherTable_Free(B);       // Free Butcher table
  MRIStepCoupling_Free(C);          // Free coupling coefficients
  SUNMatDestroy(A);                 // Free fast matrix
  SUNLinSolFree(LS);                // Free fast linear solver

  return 0;
}


static int OutputStatsMRI(void *arkode_mem, void* inner_arkode_mem)
{
  int retval;

  long int nsts, nfs, nstf, nffe, nffi, nnif, nncf, njef;

  // Get some slow integrator statistics
  retval = MRIStepGetNumSteps(arkode_mem, &nsts);
  if (check_retval(&retval, "MRIStepGetNumSteps", 1)) return 1;

  retval = MRIStepGetNumRhsEvals(arkode_mem, &nfs);
  if (check_retval(&retval, "MRIStepGetNumRhsEvals", 1)) return 1;

  // Get some fast integrator statistics
  retval = ARKStepGetNumSteps(inner_arkode_mem, &nstf);
  if (check_retval(&retval, "ARKStepGetNumSteps", 1)) return 1;

  retval = ARKStepGetNumRhsEvals(inner_arkode_mem, &nffe, &nffi);
  if (check_retval(&retval, "ARKStepGetNumRhsEvals", 1)) return 1;

  // Print some final statistics
  cout << endl;
  cout << "Final Solver Statistics:" << endl;
  cout << "  Slow Steps = " << nsts << endl;
  cout << "  Fast Steps = " << nstf << endl;
  cout << "  Slow Rhs evals = " << nfs << endl;
  cout << "  Fast Rhs evals = " << nffi << endl;

  // Get/print fast integrator implicit solver statistics
  retval = ARKStepGetNonlinSolvStats(inner_arkode_mem, &nnif, &nncf);
  if (check_retval(&retval, "ARKStepGetNonlinSolvStats", 1)) return 1;

  retval = ARKStepGetNumJacEvals(inner_arkode_mem, &njef);
  if (check_retval(&retval, "ARKStepGetNumJacEvals", 1)) return 1;

  cout << "  Fast Newton iters      = " << nnif << endl;
  cout << "  Fast Newton conv fails = " << nncf << endl;
  cout << "  Fast Jacobian evals    = " << njef << endl;
  cout << endl;

  return 0;
}


// -----------------------------------------------------------------------------
// Evolve with Lie-Trotter Splitting
// -----------------------------------------------------------------------------


static int EvolveLT(N_Vector y, UserData *udata, UserOptions *uopts,
                    OutputData *outdata)
{
  // Reusable error flag
  int retval;

  // -------------------------
  // Setup the fast integrator
  // -------------------------

  // Implicit reactions
  void *inner_arkode_mem = ARKStepCreate(NULL, RhsReaction, uopts->T0, y);
  if (check_retval((void *) inner_arkode_mem, "ARKStepCreate", 0)) return 1;

  // Set method order to use
  retval = ARKStepSetOrder(inner_arkode_mem, 2);
  if (check_retval(&retval, "ARKStepSetOrder",1)) return 1;

  // Set the fast step size
  retval = ARKStepSetFixedStep(inner_arkode_mem, uopts->hf);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Specify fast tolerances
  retval = ARKStepSStolerances(inner_arkode_mem, uopts->reltol, uopts->abstol);
  if (check_retval(&retval, "ARKStepSStolerances", 1)) return 1;

  // Initialize matrix and linear solver data structures
  SUNMatrix A = SUNBandMatrix(udata->NEQ, 3, 3);
  if (check_retval((void *)A, "SUNBandMatrix", 0)) return 1;

  SUNLinearSolver LS = SUNLinSol_Band(y, A);
  if (check_retval((void *)LS, "SUNLinSol_Band", 0)) return 1;

  // Attach matrix and linear solver
  retval = ARKStepSetLinearSolver(inner_arkode_mem, LS, A);
  if (check_retval(&retval, "ARKStepSetLinearSolver", 1)) return 1;

  // Set the Jacobian routine
  retval = ARKStepSetJacFn(inner_arkode_mem, JacReaction);
  if (check_retval(&retval, "ARKStepSetJacFn", 1)) return 1;

  // Set max number of nonlinear iters
  retval = ARKStepSetMaxNonlinIters(inner_arkode_mem, 10);
  if (check_retval(&retval, "ARKStepSetMaxNonlinIters", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(inner_arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // -------------------------
  // Setup the slow integrator
  // -------------------------

  // Integrator data and settings
  void *arkode_mem = ARKStepCreate(RhsAdvection, NULL, uopts->T0, y);
  if (check_retval((void *) arkode_mem, "ARKStepCreate", 0)) return 1;

  // attach expicit Euler
  ARKodeButcherTable Bs = ARKodeButcherTable_Alloc(1, SUNFALSE);
  if (check_retval((void *)Bs, "ARKodeButcherTable_Alloc", 0)) return 1;

  Bs->A[0][0] = ZERO;
  Bs->b[0]    = ONE;
  Bs->c[0]    = ZERO;
  Bs->q       = 1;

  retval = ARKStepSetTables(arkode_mem, 1, 0, NULL, Bs);
  if (check_retval(&retval, "ARKStepSetTables", 1)) return 1;

  // Set the step size
  retval = ARKStepSetFixedStep(arkode_mem, uopts->hs);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // Set maximum number of steps taken by solver
  retval = MRIStepSetMaxNumSteps(arkode_mem, 100000000);
  if (check_retval(&retval, "MRIStepSetMaxNumSteps", 1)) return 1;

  // -------------
  // Integrate ODE
  // -------------

  // Open output files
  retval = OpenOutput(y, outdata);
  if (check_retval(&retval, "OpenOutput", 1)) return 1;

  // Set initial time and first output time
  realtype t    = uopts->T0;
  realtype tout = uopts->T0 + outdata->dTout;

  // Output the initial condition
  retval = WriteOutput(t, y, outdata);
  if (check_retval(&retval, "WriteOutput", 1)) return 1;

  // Main time-stepping
  Timer evolve;

  for (int iout = 0; iout < outdata->Nt; iout++)
  {
    // Advance in time
    evolve.start();
    retval = LTStepEvolve(arkode_mem, inner_arkode_mem, uopts->hs, outdata->dTout,
                          tout, y, &t);
    evolve.stop();
    if (check_retval(&retval, "LTStepEvolve", 1)) break;

    // Write output
    retval = WriteOutput(t, y, outdata);
    if (check_retval(&retval, "WriteOutput", 1)) break;

    // Update output time
    tout += outdata->dTout;
    tout = (tout > uopts->Tf) ? uopts->Tf : tout;
  }

  // Close output
  retval = CloseOutput(outdata);
  if (check_retval(&retval, "CloseOutput", 1)) return 1;

  // --------
  // Finalize
  // --------

  // Output integration stats
  retval = OutputStatsLT(arkode_mem, inner_arkode_mem);
  if (check_retval(&retval, "OutputStats", 1)) return 1;

  cout << "Timing:" << endl;
  cout << "  Evolution: " << evolve.total() << endl;

  // Clean up
  ARKStepFree(&arkode_mem);    // Free integrator memory
  ARKodeButcherTable_Free(Bs); // Free Butcher table
  SUNMatDestroy(A);            // Free fast matrix
  SUNLinSolFree(LS);           // Free fast linear solver

  return 0;
}


static int LTStepEvolve(void *arkode_mem, void *inner_arkode_mem, realtype hs,
                        realtype dTout, realtype tout, N_Vector y, realtype *t)
{
  int      retval;
  realtype tmp_t1 = *t;
  realtype tmp_t2;

  // Stop outer method at the output time
  retval = ARKStepSetStopTime(arkode_mem, tout);
  if (check_retval(&retval, "ARKStepSetStopTime", 1)) return 1;

  // Step until tout is reached (assume hs evenly divides dTout)
  for (int i = 0; i < dTout / hs; i++)
  {
    // Outer: single step to t_n + h_s (tmp_t1 -> tmp_t2)
    retval = ARKStepEvolve(arkode_mem, tout, y, &tmp_t2, ARK_ONE_STEP);
    if (check_retval(&retval, "Outer: ARKStepEvolve", 1)) break;

    // Inner: reset to outer state but at t_n (tmp_t1)
    retval = ARKStepReset(inner_arkode_mem, tmp_t1, y);
    if (check_retval(&retval, "Inner: ARKStepReset", 1)) break;

    // Inner: stop at t_n + h_s (tmp_t1 -> tmp_t2)
    retval = ARKStepSetStopTime(inner_arkode_mem, tmp_t2);
    if (check_retval(&retval, "Inner: ARKStepSetStopTime", 1)) return 1;

    // Inner: subcycle to t_n + h_s (tmp_t1 -> tmp_t2, update tmp_t1)
    retval = ARKStepEvolve(inner_arkode_mem, tmp_t2, y, &tmp_t1, ARK_NORMAL);
    if (check_retval(&retval, "Inner: ARKStepEvolve", 1)) break;

    // Outer: reset to inner state at tmp_t1 (same as tmp_t2)
    retval = ARKStepReset(arkode_mem, tmp_t1, y);
    if (check_retval(&retval, "Outer: ARKStepReset", 1)) break;
  }

  // Update return time
  *t = tmp_t1;

  return retval;
}


static int OutputStatsLT(void *arkode_mem, void* inner_arkode_mem)
{
  int retval;

  long int nsts, nfs, nstf, nffe, nffi, nnif, nncf, njef;

  // Get some slow integrator statistics
  retval = MRIStepGetNumSteps(arkode_mem, &nsts);
  if (check_retval(&retval, "MRIStepGetNumSteps", 1)) return 1;

  retval = MRIStepGetNumRhsEvals(arkode_mem, &nfs);
  if (check_retval(&retval, "MRIStepGetNumRhsEvals", 1)) return 1;

  // Get some fast integrator statistics
  retval = ARKStepGetNumSteps(inner_arkode_mem, &nstf);
  if (check_retval(&retval, "ARKStepGetNumSteps", 1)) return 1;

  retval = ARKStepGetNumRhsEvals(inner_arkode_mem, &nffe, &nffi);
  if (check_retval(&retval, "ARKStepGetNumRhsEvals", 1)) return 1;

  // Print some final statistics
  cout << endl;
  cout << "Final Solver Statistics:" << endl;
  cout << "  Slow Steps = " << nsts << endl;
  cout << "  Fast Steps = " << nstf << endl;
  cout << "  Slow Rhs evals = " << nfs << endl;
  cout << "  Fast Rhs evals = " << nffi << endl;

  // Get/print fast integrator implicit solver statistics
  retval = ARKStepGetNonlinSolvStats(inner_arkode_mem, &nnif, &nncf);
  if (check_retval(&retval, "ARKStepGetNonlinSolvStats", 1)) return 1;

  retval = ARKStepGetNumJacEvals(inner_arkode_mem, &njef);
  if (check_retval(&retval, "ARKStepGetNumJacEvals", 1)) return 1;

  cout << "  Fast Newton iters      = " << nnif << endl;
  cout << "  Fast Newton conv fails = " << nncf << endl;
  cout << "  Fast Jacobian evals    = " << njef << endl;
  cout << endl;

  return 0;
}


// -----------------------------------------------------------------------------
// Evolve with Strang-Marchuk Splitting
// -----------------------------------------------------------------------------


static int EvolveSM(N_Vector y, UserData *udata, UserOptions *uopts,
                    OutputData *outdata)
{
  // Reusable error flag
  int retval;

  // -------------------------
  // Setup the fast integrator
  // -------------------------

  // Implicit reactions
  void *inner_arkode_mem = ARKStepCreate(NULL, RhsReaction, uopts->T0, y);
  if (check_retval((void *) inner_arkode_mem, "ARKStepCreate", 0)) return 1;

  // Set method order to use
  retval = ARKStepSetOrder(inner_arkode_mem, 2);
  if (check_retval(&retval, "ARKStepSetOrder",1)) return 1;

  // Set the fast step size
  retval = ARKStepSetFixedStep(inner_arkode_mem, uopts->hf);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Specify fast tolerances
  retval = ARKStepSStolerances(inner_arkode_mem, uopts->reltol, uopts->abstol);
  if (check_retval(&retval, "ARKStepSStolerances", 1)) return 1;

  // Initialize matrix and linear solver data structures
  SUNMatrix A = SUNBandMatrix(udata->NEQ, 3, 3);
  if (check_retval((void *)A, "SUNBandMatrix", 0)) return 1;

  SUNLinearSolver LS = SUNLinSol_Band(y, A);
  if (check_retval((void *)LS, "SUNLinSol_Band", 0)) return 1;

  // Attach matrix and linear solver
  retval = ARKStepSetLinearSolver(inner_arkode_mem, LS, A);
  if (check_retval(&retval, "ARKStepSetLinearSolver", 1)) return 1;

  // Set the Jacobian routine
  retval = ARKStepSetJacFn(inner_arkode_mem, JacReaction);
  if (check_retval(&retval, "ARKStepSetJacFn", 1)) return 1;

  // Set max number of nonlinear iters
  retval = ARKStepSetMaxNonlinIters(inner_arkode_mem, 10);
  if (check_retval(&retval, "ARKStepSetMaxNonlinIters", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(inner_arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // -------------------------
  // Setup the slow integrator
  // -------------------------

  // Integrator data and settings
  void *arkode_mem = ARKStepCreate(RhsAdvection, NULL, uopts->T0, y);
  if (check_retval((void *) arkode_mem, "ARKStepCreate", 0)) return 1;

  // Set method order to use
  retval = ARKStepSetOrder(arkode_mem, 2);
  if (check_retval(&retval, "ARKStepSetOrder",1)) return 1;

  // Set the step size
  retval = ARKStepSetFixedStep(arkode_mem, uopts->hs / TWO);
  if (check_retval(&retval, "ARKStepSetFixedStep", 1)) return 1;

  // Attach user data to fast integrator
  retval = ARKStepSetUserData(arkode_mem, (void *) udata);
  if (check_retval(&retval, "ARKStepSetUserData", 1)) return 1;

  // Set maximum number of steps taken by solver
  retval = MRIStepSetMaxNumSteps(arkode_mem, 100000000);
  if (check_retval(&retval, "MRIStepSetMaxNumSteps", 1)) return 1;

  // -------------
  // Integrate ODE
  // -------------

  // Open output files
  retval = OpenOutput(y, outdata);
  if (check_retval(&retval, "OpenOutput", 1)) return 1;

  // Set initial time and first output time
  realtype t    = uopts->T0;
  realtype tout = uopts->T0 + outdata->dTout;

  // Output the initial condition
  retval = WriteOutput(t, y, outdata);
  if (check_retval(&retval, "WriteOutput", 1)) return 1;

  // Main time-stepping
  Timer evolve;

  for (int iout = 0; iout < outdata->Nt; iout++)
  {
    // Advance in time
    evolve.start();
    retval = SMStepEvolve(arkode_mem, inner_arkode_mem, uopts->hs,
                          outdata->dTout, tout, y, &t);
    evolve.stop();
    if (check_retval(&retval, "SMStepEvolve", 1)) break;

    // Write output
    retval = WriteOutput(t, y, outdata);
    if (check_retval(&retval, "WriteOutput", 1)) break;

    // Update output time
    tout += outdata->dTout;
    tout = (tout > uopts->Tf) ? uopts->Tf : tout;
  }

  // Close output
  retval = CloseOutput(outdata);
  if (check_retval(&retval, "CloseOutput", 1)) return 1;

  // --------
  // Finalize
  // --------

  // Output integration stats
  retval = OutputStatsSM(arkode_mem, inner_arkode_mem);
  if (check_retval(&retval, "OutputStats", 1)) return 1;

  cout << "Timing:" << endl;
  cout << "  Evolution: " << evolve.total() << endl;

  // Clean up
  ARKStepFree(&arkode_mem);  // Free integrator memory
  SUNMatDestroy(A);          // Free fast matrix
  SUNLinSolFree(LS);         // Free fast linear solver

  return 0;
}


static int SMStepEvolve(void *arkode_mem, void *inner_arkode_mem, realtype hs,
                        realtype dTout, realtype tout, N_Vector y, realtype *t)
{
  int      retval;
  realtype tmp_t1 = *t;
  realtype tmp_t2, tmp_t3;

  // Stop outer method at the output time
  retval = ARKStepSetStopTime(arkode_mem, tout);
  if (check_retval(&retval, "Outer: ARKStepSetStopTime", 1)) return 1;

  // Step until tout is reached (assume hs evenly divides dTout)
  for (int i = 0; i < dTout / hs; i++)
  {
    // Outer: single step to t_n + h_s / 2 (tmp_t1 -> tmp_t2)
    retval = ARKStepEvolve(arkode_mem, tout, y, &tmp_t2, ARK_ONE_STEP);
    if (check_retval(&retval, "Outer 1: ARKStepEvolve", 1)) break;

    // Inner: reset to outer state but at t_n (tmp_t1)
    retval = ARKStepReset(inner_arkode_mem, tmp_t1, y);
    if (check_retval(&retval, "Inner: ARKStepReset", 1)) break;

    // Inner: stop at t_n + h_s (tmp_t1 -> tmp_t3)
    tmp_t3 = tmp_t1 + TWO * (tmp_t2 - tmp_t1);
    retval = ARKStepSetStopTime(inner_arkode_mem, tmp_t3);
    if (check_retval(&retval, "Inner: ARKStepSetStopTime", 1)) return 1;

    // Inner: subcycle to t_n + h_s (tmp_t1 -> tmp_t3, update tmp_t1)
    retval = ARKStepEvolve(inner_arkode_mem, tmp_t3, y, &tmp_t1, ARK_NORMAL);
    if (check_retval(&retval, "Inner: ARKStepEvolve", 1)) break;

    // Outer: reset to inner state but at t_n + h_s / 2 (tmp_t2)
    retval = ARKStepReset(arkode_mem, tmp_t2, y);
    if (check_retval(&retval, "Outer: ARKStepReset", 1)) break;

    // Outer: single step to t_n + h_s (tmp_t2 -> tmp_t1, same as tmp_t3)
    retval = ARKStepEvolve(arkode_mem, tmp_t1, y, &tmp_t2, ARK_ONE_STEP);
    if (check_retval(&retval, "Outer: ARKStepEvolve", 1)) break;
  }

  // Update return time
  *t = tmp_t2;

  return retval;
}


static int OutputStatsSM(void *arkode_mem, void* inner_arkode_mem)
{
  int retval;

  long int nsts, nfs, nstf, nffe, nffi, nnif, nncf, njef;

  // Get some slow integrator statistics
  retval = MRIStepGetNumSteps(arkode_mem, &nsts);
  if (check_retval(&retval, "MRIStepGetNumSteps", 1)) return 1;

  retval = MRIStepGetNumRhsEvals(arkode_mem, &nfs);
  if (check_retval(&retval, "MRIStepGetNumRhsEvals", 1)) return 1;

  // Get some fast integrator statistics
  retval = ARKStepGetNumSteps(inner_arkode_mem, &nstf);
  if (check_retval(&retval, "ARKStepGetNumSteps", 1)) return 1;

  retval = ARKStepGetNumRhsEvals(inner_arkode_mem, &nffe, &nffi);
  if (check_retval(&retval, "ARKStepGetNumRhsEvals", 1)) return 1;

  // Print some final statistics
  cout << endl;
  cout << "Final Solver Statistics:" << endl;
  cout << "  Slow Steps = " << nsts << endl;
  cout << "  Fast Steps = " << nstf << endl;
  cout << "  Slow Rhs evals = " << nfs << endl;
  cout << "  Fast Rhs evals = " << nffi << endl;

  // Get/print fast integrator implicit solver statistics
  retval = ARKStepGetNonlinSolvStats(inner_arkode_mem, &nnif, &nncf);
  if (check_retval(&retval, "ARKStepGetNonlinSolvStats", 1)) return 1;

  retval = ARKStepGetNumJacEvals(inner_arkode_mem, &njef);
  if (check_retval(&retval, "ARKStepGetNumJacEvals", 1)) return 1;

  cout << "  Fast Newton iters      = " << nnif << endl;
  cout << "  Fast Newton conv fails = " << nncf << endl;
  cout << "  Fast Jacobian evals    = " << njef << endl;
  cout << endl;

  return 0;
}


// -----------------------------------------------------------------------------
// Functions called by the integrator
// -----------------------------------------------------------------------------


// Compute the advection ODE Rhs
static int RhsAdvection(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  // Access problem data
  UserData     *udata = (UserData*) user_data;
  sunindextype N      = udata->N;

  // Access data arrays
  realtype *Ydata = N_VGetArrayPointer(y);
  if (check_retval((void *)Ydata, "N_VGetArrayPointer", 0)) return 1;

  realtype *dYdata = N_VGetArrayPointer(ydot);
  if (check_retval((void *)dYdata, "N_VGetArrayPointer", 0)) return 1;

  // Set advection constants
  realtype auconst = -udata->au / RCONST(2.0) / udata->dx;
  realtype avconst = -udata->av / RCONST(2.0) / udata->dx;
  realtype awconst = -udata->aw / RCONST(2.0) / udata->dx;

  // Enforce left stationary boundary condition
  dYdata[IDX(0,0)] = dYdata[IDX(0,1)] = dYdata[IDX(0,2)] = ZERO;

  // Compute reaction at interior nodes
  realtype ul, ur, vl, vr, wl, wr;

  for (sunindextype i = 1; i < N - 1; i++)
  {
    ul = Ydata[IDX(i-1,0)];  ur = Ydata[IDX(i+1,0)];
    vl = Ydata[IDX(i-1,1)];  vr = Ydata[IDX(i+1,1)];
    wl = Ydata[IDX(i-1,2)];  wr = Ydata[IDX(i+1,2)];

    dYdata[IDX(i,0)] = (ur - ul) * auconst;
    dYdata[IDX(i,1)] = (vr - vl) * avconst;
    dYdata[IDX(i,2)] = (wr - wl) * awconst;
  }

  // Enforce right stationary boundary condition
  dYdata[IDX(N-1,0)] = dYdata[IDX(N-1,1)] = dYdata[IDX(N-1,2)] = ZERO;

  // Return with success
  return 0;
}


// Compute the reaction ODE Rhs
static int RhsReaction(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  // Access problem data
  UserData *udata = (UserData*) user_data;

  // Set variable shortcuts
  sunindextype N = udata->N;
  realtype     a  = udata->a;
  realtype     b  = udata->b;
  realtype     ep = udata->ep;

  // Access data arrays
  realtype *Ydata = N_VGetArrayPointer(y);
  if (check_retval((void *)Ydata, "N_VGetArrayPointer", 0)) return 1;

  realtype *dYdata = N_VGetArrayPointer(ydot);
  if (check_retval((void *)dYdata, "N_VGetArrayPointer", 0)) return 1;

  // Enforce left stationary boundary condition
  dYdata[IDX(0,0)] = dYdata[IDX(0,1)] = dYdata[IDX(0,2)] = ZERO;

  // Iterate over interior domain
  realtype u, v, w;

  for (sunindextype i = 1; i < N - 1; i++)
  {
    u = Ydata[IDX(i,0)];
    v = Ydata[IDX(i,1)];
    w = Ydata[IDX(i,2)];

    dYdata[IDX(i,0)] = a - (w + ONE) * u + v * u * u;
    dYdata[IDX(i,1)] = w * u - v * u * u;
    dYdata[IDX(i,2)] = ((b - w) / ep) - w * u;
  }

  // Enforce right stationary boundary condition
  dYdata[IDX(N-1,0)] = dYdata[IDX(N-1,1)] = dYdata[IDX(N-1,2)] = ZERO;

  return 0;
}


// Compute the reaction Jacobian
static int JacReaction(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                       void *user_data, N_Vector tmp1, N_Vector tmp2,
                       N_Vector tmp3)
{
  // Access problem data
  UserData *udata = (UserData*) user_data;

  // Set shortcuts
  sunindextype N  = udata->N;
  realtype     ep = udata->ep;

  // Access solution array
  realtype *Ydata = N_VGetArrayPointer(y);
  if (check_retval((void *)Ydata, "N_VGetArrayPointer", 0)) return 1;

  // Iterate over interior nodes
  realtype u, v, w;

  for (sunindextype i = 1; i < N - 1; i++)
  {
    // Set nodal value shortcuts
    u = Ydata[IDX(i,0)];
    v = Ydata[IDX(i,1)];
    w = Ydata[IDX(i,2)];

    // All vars wrt u
    SM_ELEMENT_B(J,IDX(i,0),IDX(i,0)) = TWO * u * v - (w + ONE);
    SM_ELEMENT_B(J,IDX(i,1),IDX(i,0)) = w - TWO * u * v;
    SM_ELEMENT_B(J,IDX(i,2),IDX(i,0)) = -w;

    // All vars wrt v
    SM_ELEMENT_B(J,IDX(i,0),IDX(i,1)) = u * u;
    SM_ELEMENT_B(J,IDX(i,1),IDX(i,1)) = -u * u;

    // All vars wrt w
    SM_ELEMENT_B(J,IDX(i,0),IDX(i,2)) = -u;
    SM_ELEMENT_B(J,IDX(i,1),IDX(i,2)) = u;
    SM_ELEMENT_B(J,IDX(i,2),IDX(i,2)) = (-ONE / ep) - u;
  }

  // Return with success
  return 0;
}


// -----------------------------------------------------------------------------
// Output and utility functions
// -----------------------------------------------------------------------------


// Set the initial condition
static int SetIC(N_Vector y, void *user_data)
{
  // Access problem data
  UserData *udata = (UserData*) user_data;

  // set variable shortcuts
  sunindextype N  = udata->N;
  realtype     a  = udata->a;
  realtype     b  = udata->b;
  realtype     dx = udata->dx;

  // Access data array from NVector y
  realtype *Ydata = N_VGetArrayPointer(y);
  if (check_retval((void *)Ydata, "N_VGetArrayPointer", 0)) return 1;

  // Set initial conditions into y
  for (sunindextype i = 0; i < N; i++)
  {
    Ydata[IDX(i,0)] =  a  + RCONST(0.1) * sin(PI * i * dx);  // u
    Ydata[IDX(i,1)] = b/a + RCONST(0.1) * sin(PI * i * dx);  // v
    Ydata[IDX(i,2)] =  b  + RCONST(0.1) * sin(PI * i * dx);  // w
  }

  // Return with success
  return 0;
}


// Open output stream, allocate data
static int OpenOutput(N_Vector y, OutputData *outdata)
{
  // Create vector masks
  outdata->umask = N_VClone(y);
  if (check_retval((void *)(outdata->umask), "N_VNew_Serial", 0)) return 1;

  outdata->vmask = N_VClone(y);
  if (check_retval((void *)(outdata->vmask), "N_VNew_Serial", 0)) return 1;

  outdata->wmask = N_VClone(y);
  if (check_retval((void *)(outdata->wmask), "N_VNew_Serial", 0)) return 1;

  // Set mask array values for each solution component
  realtype *data = NULL;

  N_VConst(ZERO, outdata->umask);
  data = N_VGetArrayPointer(outdata->umask);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return 1;
  for (sunindextype i = 0; i < outdata->N; i++) data[IDX(i,0)] = ONE;

  N_VConst(ZERO, outdata->vmask);
  data = N_VGetArrayPointer(outdata->vmask);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return 1;
  for (sunindextype i = 0; i < outdata->N; i++) data[IDX(i,1)] = ONE;

  N_VConst(ZERO, outdata->wmask);
  data = N_VGetArrayPointer(outdata->wmask);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return 1;
  for (sunindextype i = 0; i < outdata->N; i++) data[IDX(i,2)] = ONE;

  // Open output streams for solution
  outdata->yout.open("advection_reaction_1D_mri.out");
  outdata->yout <<  "# title Advection-Reaction (Brusselator)" << endl;
  outdata->yout <<  "# nvar 3" << endl;
  outdata->yout <<  "# vars u v w" << endl;
  outdata->yout <<  "# nx " << outdata->N << endl;
  outdata->yout <<  "# xl " << 0.0 << endl;
  outdata->yout <<  "# xu " << 1.0 << endl;
  outdata->yout << scientific;
  outdata->yout << setprecision(numeric_limits<realtype>::digits10);

  // Print output to screen
  cout << "     t         ||u||_rms     ||v||_rms     ||w||_rms" << endl;
  cout << "------------------------------------------------------" << endl;
  cout << scientific << setprecision(4);

  return 0;
}


// Write output to screen and file
static int WriteOutput(realtype t, N_Vector y, OutputData *outdata)
{
  // Print solution norms to screen
  realtype u = N_VWL2Norm(y, outdata->umask);
  u = sqrt(u * u / outdata->N);

  realtype v = N_VWL2Norm(y, outdata->vmask);
  v = sqrt(v * v / outdata->N);

  realtype w = N_VWL2Norm(y, outdata->wmask);
  w = sqrt(w * w / outdata->N);

  cout << setw(11) << t << setw(14) << u << setw(14) << v << setw(14) << w
       << endl;

  // Get solution data array
  realtype *data = N_VGetArrayPointer(y);
  if (check_retval((void *)data, "N_VGetArrayPointer", 0)) return 1;

  // Write solution to disk
  outdata->yout << t << " ";
  for (sunindextype i = 0; i < outdata->NEQ; i++)
  {
    outdata->yout << data[i] << " ";
  }
  outdata->yout << endl;

  return 0;
}


// Close output stream, free data
static int CloseOutput(OutputData *outdata)
{
  cout << "------------------------------------------------------" << endl;
  outdata->yout.close();

  N_VDestroy(outdata->umask);
  N_VDestroy(outdata->vmask);
  N_VDestroy(outdata->wmask);

  return 0;
}


// -----------------------------------------------------------------------------
// Private helper functions
// -----------------------------------------------------------------------------


// Read command line inputs
static int ReadInputs(int *argc, char ***argv, UserData *udata,
                      UserOptions *uopts, OutputData *outdata)
{
  // Check for input args
  int arg_idx = 1;

  while (arg_idx < (*argc))
  {
    string arg = (*argv)[arg_idx++];

    if (arg == "--mesh")
    {
      udata->N = stoi((*argv)[arg_idx++]);
      if (udata->N < 3)
      {
        cerr << "ERROR: number of mesh points must be >= 3" << endl;
        return -1;
      }
    }
    else if (arg == "--Tf")
    {
      uopts->Tf = stod((*argv)[arg_idx++]);
      if (uopts->Tf <= ZERO)
      {
        cerr << "ERROR: Tf must be > 0" << endl;
        return -1;
      }
    }
    else if (arg == "--Nt")
    {
      outdata->Nt = stoi((*argv)[arg_idx++]);
      if (outdata->Nt < 1)
      {
        cerr << "ERROR: Nt must be > 0" << endl;
        return -1;
      }
    }
    else if (arg == "--method")
    {
      uopts->method = stoi((*argv)[arg_idx++]);
      if (uopts->method < 0)
      {
        cerr << "ERROR: method must be >= 0" << endl;
        return -1;
      }
    }
    else if (arg == "--order")
    {
      uopts->order = stoi((*argv)[arg_idx++]);
      if (uopts->order < 2 || uopts->order > 4)
      {
        cerr << "ERROR: order must be > 1 and < 5" << endl;
        return -1;
      }
    }
    else if (arg == "--m")
    {
      uopts->m = stoi((*argv)[arg_idx++]);
      if (uopts->m < 1)
      {
        cerr << "ERROR: m must be > 0" << endl;
        return -1;
      }
    }
    else if (arg == "--hs")
    {
      uopts->hs = stod((*argv)[arg_idx++]);
      if (uopts->hs < ZERO)
      {
        cerr << "ERROR: hs must be > 0" << endl;
        return -1;
      }
    }
    else if (arg == "--rtol")
    {
      uopts->reltol = stod((*argv)[arg_idx++]);
      if (uopts->reltol < ZERO)
      {
        cerr << "ERROR: rtol must be > 0" << endl;
        return -1;
      }
    }
    else if (arg == "--atol")
    {
      uopts->abstol = stod((*argv)[arg_idx++]);
      if (uopts->abstol < ZERO)
      {
        cerr << "ERROR: atol must be > 0" << endl;
        return -1;
      }
    }
    // Help
    // else if (arg == "--help")
    // {
    //   InputHelp();
    //   return -1;
    // }
    // Unknown input
    else
    {
      cerr << "ERROR: Invalid input " << arg << endl;
      // InputHelp();
      return -1;
    }
  }

  // Recompute total number of unknowns, mesh spacing, and hf
  udata->NEQ = 3 * udata->N;
  udata->dx  = ONE / (udata->N - 1);
  uopts->hf  = uopts->hs / uopts->m;

  // Compute time between outputs and save problem sizes
  outdata->dTout = (uopts->Tf - uopts->T0) / outdata->Nt;
  outdata->N     = udata->N;
  outdata->NEQ   = udata->NEQ;

  // Return success
  return 0;
}


// Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *errvalue;

  // Check if SUNDIALS function returned NULL pointer - no memory allocated
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  // Check if flag < 0
  else if (opt == 1) {
    errvalue = (int *) returnvalue;
    if (*errvalue < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errvalue);
      return 1;
    }
  }

  // Check if function returned NULL pointer - no memory allocated
  else if (opt == 2 && returnvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}

//---- end of file ----
