/*-----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2022, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem with analytical
 * solution,
 *    dy/dt = lamda*y + 1/(1+t^2) - lamda*atan(t)
 * for t in the interval [0.0, 10.0], with initial condition: y=0.
 *
 * The stiffness of the problem is directly proportional to the
 * value of "lamda".  The value of lamda should be negative to
 * result in a well-posed ODE; for values with magnitude larger
 * than 100 the problem becomes quite stiff.
 *
 * This program solves the problem with the DIRK method,
 * Newton iteration with the dense SUNLinearSolver, and a
 * user-supplied Jacobian routine.
 * Output is printed every 1.0 units of time (10 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

/* Header files */
#include <stdio.h>
#include <math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype* ydata = N_VGetArrayPointer(y);
  realtype* fdata = N_VGetArrayPointer(ydot);

  fdata[0] = -50.0 * (ydata[0] - cos(t));

  return 0;
}

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  realtype *Jdata = SUNDenseMatrix_Data(J);

  Jdata[0] = -50.0;

  return 0;
}

int main(int argc, char** argv)
{
  realtype T0 = RCONST(0.0);
  realtype Tf = RCONST(1.5);

  realtype reltol = RCONST(5.0e-2);
  realtype abstol = RCONST(5.0e-2);
  if (argc > 2)
  {
    reltol = atof(argv[1]);
    abstol = atof(argv[2]);
  }
  printf("reltol %g\n", reltol);
  printf("abstol %g\n", abstol);

  /* Create the SUNDIALS context object for this simulation */
  SUNContext ctx;
  SUNContext_Create(NULL, &ctx);

  /* Initialize data structures */
  N_Vector y = N_VNew_Serial(1, ctx);
  SUNMatrix A = SUNDenseMatrix(1, 1, ctx);
  SUNLinearSolver LS = SUNLinSol_Dense(y, A, ctx);

  realtype* ydata = N_VGetArrayPointer(y);
  ydata[0] = RCONST(0.0);

  void* cvode_mem = CVodeCreate(CV_BDF, ctx);
  CVodeInit(cvode_mem, f, T0, y);
  CVodeSStolerances(cvode_mem, reltol, abstol);
  CVodeSetLinearSolver(cvode_mem, LS, A);
  CVodeSetJacFn(cvode_mem, Jac);
  // CVodeSetInitStep(cvode_mem, 0.00001);

  FILE* UFID = fopen("solution_cvode.txt","w");
  fprintf(UFID,"# t u\n");
  fprintf(UFID," %.16"ESYM" %.16"ESYM"\n", T0, ydata[0]);

  /*  */
  realtype t = T0;
  printf("        t           u\n");
  printf("   ---------------------\n");
  while (t <= Tf)
  {
    CVode(cvode_mem, Tf, y, &t, CV_ONE_STEP);

    printf("  %10.6"FSYM"  %10.6"FSYM"\n", t, ydata[0]);
    fprintf(UFID," %.16"ESYM" %.16"ESYM"\n", t, ydata[0]);
  }
  printf("   ---------------------\n");
  fclose(UFID);
  CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);

  /* Print analytic solution */
  UFID = fopen("solution_analytic.txt","w");
  int npts = 10000;
  t = 0;
  int i = 0;
  while (t < Tf)
  {
    realtype ytrue = (50.0 * (-50.0 * exp(-50.0 * t) + sin(t) + 50.0 * cos(t))) / 2501.0;
    fprintf(UFID," %.16"ESYM" %.16"ESYM"\n", t, ytrue);
    t += i * (Tf - T0) / npts;
    i++;
  }
  fclose(UFID);

  /* Clean up and return */
  N_VDestroy(y);
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  SUNContext_Free(&ctx);

  return 0;
}
