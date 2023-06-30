/* clang-format: off */
/* ----------------------------------------------------------------------------
 * Programmer(s): Cody J. Balos @ LLNL
 * ----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ----------------------------------------------------------------------------
 * In this example we consider the anharmonic oscillator
 *    q'(t) = p(t)
 *    p'(t) = -( omega^2(t)*q^2 + 3*a(t)*q^2 + 4*b(t)*q^3 )
 * With the initial conditions q(0) = 1, p(0) = 0.
 * The Hamiltonian for the system is
 *    H(p,q,t) = 1/2*p^2 + 1/2*omega^2(t)*q^2 + a(t)*q^3 + b(t)*q^4
 * where omega(t) = cos(t/2), a(t) = 0.05*sin(t/3), b(t) = 0.08*cos^2(t/3).
 * We simulate the problem on t = [0, 30] using the symplectic methods in
 * SPRKStep.
 *
 * This is example 7.2 from:
 * Struckmeier, J., & Riedel, C. (2002). Canonical transformations and exact
 * invariants for time‐dependent Hamiltonian systems. Annalen der Physik, 11(1),
 * 15-38.
 *
 * The example has the following command line arguments:
 *   --order <int>               the order of the method to use (default 4)
 *   --dt <Real>                 the fixed-time step size to use (default 0.01)
 *   --nout <int>                the number of output times (default 100)
 *   --use-compensated-sums      turns on compensated summation in ARKODE where
 *                               applicable
 * --------------------------------------------------------------------------*/
/* clang-format: on */

#include <arkode/arkode_sprk.h>
#include <arkode/arkode_sprkstep.h> /* prototypes for SPRKStep fcts., consts */
#include <math.h>
#include <nvector/nvector_serial.h> /* serial N_Vector type, fcts., macros  */
#include <stdio.h>
#include <string.h>
#include <sundials/sundials_math.h> /* def. math fcns, 'sunrealtype'           */
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include "arkode/arkode.h"

typedef struct
{
  int order;
  int num_output_times;
  int use_compsums;
  sunrealtype dt;
} ProgramArgs;

/* RHS functions */
static int pdot(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
static int qdot(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);

/* Helper functions */
static sunrealtype Hamiltonian(N_Vector yvec, sunrealtype t);
static int ParseArgs(int argc, char* argv[], ProgramArgs* args);
static void PrintHelp();
static int check_retval(void* returnvalue, const char* funcname, int opt);

int main(int argc, char* argv[])
{
  ProgramArgs args;
  SUNContext sunctx  = NULL;
  N_Vector y         = NULL;
  sunrealtype* ydata = NULL;
  sunrealtype tout   = NAN;
  sunrealtype tret   = NAN;
  void* arkode_mem   = NULL;
  int iout           = 0;
  int retval         = 0;

  /* Parse the command line arguments */
  if (ParseArgs(argc, argv, &args)) { return 1; };

  /* Default integrator options */
  int order                  = args.order;
  int use_compsums           = args.use_compsums;
  const int num_output_times = args.num_output_times;

  /* Default problem parameters */
  const sunrealtype T0    = SUN_RCONST(0.0);
  sunrealtype Tf          = SUN_RCONST(30.0);
  sunrealtype dt          = args.dt;
  const sunrealtype dTout = (Tf - T0) / ((sunrealtype)num_output_times);

  /* Create the SUNDIALS context object for this simulation */
  retval = SUNContext_Create(NULL, &sunctx);
  if (check_retval(&retval, "SUNContext_Create", 1)) return 1;

  printf("\n   Begin time-dependent anharmonic oscillator problem\n\n");

  /* Allocate our state vector */
  y = N_VNew_Serial(4, sunctx);

  /* Fill the initial conditions */
  ydata    = N_VGetArrayPointer(y);
  ydata[0] = 0; /* ptau */
  ydata[1] = 0; /* \dot{q} = p */
  ydata[2] = 0; /* qtau */
  ydata[3] = 1; /* \ddot{q} = \dot{p} */

  /* Create SPRKStep integrator */
  arkode_mem = SPRKStepCreate(qdot, pdot, T0, y, sunctx);

  retval = SPRKStepSetOrder(arkode_mem, order);
  if (check_retval(&retval, "SPRKStepSetOrder", 1)) { return 1; }

  retval = SPRKStepSetUseCompensatedSums(arkode_mem, use_compsums);
  if (check_retval(&retval, "SPRKStepSetUseCompensatedSums", 1)) { return 1; }

  retval = SPRKStepSetFixedStep(arkode_mem, dt);
  if (check_retval(&retval, "SPRKStepSetFixedStep", 1)) { return 1; }

  retval = SPRKStepSetMaxNumSteps(arkode_mem, ((long int)ceil(Tf / dt)) + 2);
  if (check_retval(&retval, "SPRKStepSetMaxNumSteps", 1)) { return 1; }

  /* Print out starting Hamiltonian before integrating */
  tret = T0;
  tout = T0 + dTout;
  /* Output current integration status */
  fprintf(stdout, "t = %.6Lf, q(t) = %.6Lf, H = %.6Lf\n", tret, ydata[1],
          Hamiltonian(y, tret));

  /* Do integration */
  for (iout = 0; iout < num_output_times; iout++)
  {
    SPRKStepSetStopTime(arkode_mem, tout);
    retval = SPRKStepEvolve(arkode_mem, tout, y, &tret, ARK_NORMAL);

    /* Output current integration status */
    fprintf(stdout, "t = %.6Lf, q(t) = %.6Lf, H = %.6Lf\n", tret, ydata[1],
            Hamiltonian(y, tret));

    /* Check if the solve was successful, if so, update the time and continue */
    if (retval >= 0)
    {
      tout += dTout;
      tout = (tout > Tf) ? Tf : tout;
    }
    else
    {
      fprintf(stderr, "Solver failure, stopping integration\n");
      break;
    }
  }

  fprintf(stdout, "\n");
  SPRKStepPrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  N_VDestroy(y);
  SPRKStepFree(&arkode_mem);
  SUNContext_Free(&sunctx);

  return 0;
}

sunrealtype omega(sunrealtype t) { return cos(t / 2.0); }

sunrealtype a(sunrealtype t) { return 0.05 * sin(t / 3.0); }

sunrealtype b(sunrealtype t) { return 0.08 * cos(t / 3.0) * cos(t / 3.0); }

sunrealtype Hamiltonian(N_Vector yvec, sunrealtype t)
{
  sunrealtype E       = 0.0;
  sunrealtype* y      = N_VGetArrayPointer(yvec);
  const sunrealtype p = y[1];
  const sunrealtype q = y[3];

  E = p * p / 2 + omega(t) * omega(t) * q * q / 2 + a(t) * q * q * q +
      b(t) * q * q * q * q;

  return E;
}

int qdot(sunrealtype t, N_Vector yvec, N_Vector ydotvec, void* user_data)
{
  sunrealtype* y      = N_VGetArrayPointer(yvec);
  sunrealtype* ydot   = N_VGetArrayPointer(ydotvec);
  const sunrealtype p = y[1];

  ydot[2] = 1;
  ydot[3] = p;

  return 0;
}

int pdot(sunrealtype t, N_Vector yvec, N_Vector ydotvec, void* user_data)
{
  sunrealtype* y         = N_VGetArrayPointer(yvec);
  sunrealtype* ydot      = N_VGetArrayPointer(ydotvec);
  const sunrealtype qtau = y[2];
  const sunrealtype q    = y[3];

  ydot[0] = 1;
  ydot[1] = -omega(qtau) * omega(qtau) * q - 3 * a(qtau) * q * q -
            4 * b(qtau) * q * q * q;

  return 0;
}

int ParseArgs(int argc, char* argv[], ProgramArgs* args)
{
  args->order            = 4;
  args->num_output_times = 8;
  args->use_compsums     = 0;
  args->dt               = SUN_RCONST(1e-3);

  for (int argi = 1; argi < argc; argi++)
  {
    if (!strcmp(argv[argi], "--order"))
    {
      argi++;
      args->order = atoi(argv[argi]);
    }
    else if (!strcmp(argv[argi], "--dt"))
    {
      argi++;
      args->dt = atof(argv[argi]);
    }
    else if (!strcmp(argv[argi], "--nout"))
    {
      argi++;
      args->num_output_times = atoi(argv[argi]);
    }
    else if (!strcmp(argv[argi], "--use-compensated-sums"))
    {
      args->use_compsums = 1;
    }
    else if (!strcmp(argv[argi], "--help"))
    {
      PrintHelp();
      return 1;
    }
    else
    {
      fprintf(stderr, "ERROR: unrecognized argument %s\n", argv[argi]);
      PrintHelp();
      return 1;
    }
  }

  return 0;
}

void PrintHelp()
{
  fprintf(stderr, "ark_anharmonic_symplectic: an ARKODE example demonstrating "
                  "the SPRKStep time-stepping module solving a time-dependent "
                  "anharmonic oscillator\n");
  fprintf(stderr, "  --order <int>               the order of the method to "
                  "use (default 4)\n");
  fprintf(stderr,
          "  --dt <Real>                 the fixed-time step size to use "
          "(default 0.01)\n");
  fprintf(stderr, "  --nout <int>                the number of output times "
                  "(default 100)\n");
  fprintf(stderr,
          "  --use-compensated-sums      turns on compensated summation in "
          "ARKODE where applicable\n");
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a retval so check if
             retval < 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return 1;
  }

  /* Check if retval < 0 */
  else if (opt == 1)
  {
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nERROR: %s() failed with retval = %d\n\n", funcname,
              *retval);
      return 1;
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}