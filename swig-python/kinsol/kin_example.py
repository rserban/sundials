#!/usr/bin/env python3

"""
 -----------------------------------------------------------------
 Programmer(s): Cody J. Balos @ LLNL
 -----------------------------------------------------------------
 Based on the example kinRoberts_fp.c by Carol Woodward @ LLNL
 -----------------------------------------------------------------
 Example problem:

 The following is a simple example problem, with the coding
 needed for its solution by the accelerated fixed point solver in
 KINSOL.
 The problem is from chemical kinetics, and consists of solving
 the first time step in a Backward Euler solution for the
 following three rate equations:
    dy1/dt = -.04*y1 + 1.e4*y2*y3
    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2
    dy3/dt = 3.e2*(y2)^2
 on the interval from t = 0.0 to t = 0.1, with initial
 conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
 Run statistics (optional outputs) are printed at the end.
 -----------------------------------------------------------------
"""

import ctypes
import kinsol as kin
import numpy as np
import numba as nb
from numba import cfunc
import sys
from timeit import default_timer as timer

import faulthandler


class Problem():

  def __init__(self):
    # constants
    self.NEQ    = 3     # number of equations
    self.Y10    = 1.0   # initial y components
    self.Y20    = 1.0
    self.Y30    = 1.0
    self.TOL    = 1e-10 # function tolerance
    self.DSTEP  = 0.1   # size of the single time step used
    self.PRIORS = 2


def funcRobertsCtypes(y, y_len, g, g_len, udata):
  # constants
  Y10    = udata.Y10   # initial y components
  Y20    = udata.Y20
  Y30    = udata.Y30
  DSTEP  = udata.DSTEP # size of the single time step used

  y1 = y[0]
  y2 = y[1]
  y3 = y[2]

  yd1 = DSTEP * (-0.04*y1 + 1.0e4*y2*y3)
  yd3 = DSTEP * 3.0e2*y2*y2

  g[0] = yd1 + Y10
  g[1] = -yd1 - yd3 + Y20
  g[2] = yd3 + Y30

  return 0


def jacFuncCtypes(y, y_size, fu, J, J_size, udata, tmp1, tmp2):
  g1 = fu[0]
  g2 = fu[1]
  g3 = fu[2]

  print(g1)
  print(g2)
  print(g3)

  J[0] = -0.04
  J[1] = 1.0e4*g3
  J[2] = 1.0e4*g2

  J[3] = 0.04
  J[4] = -1.0e4*g3-(6.0e2)*g2
  J[5] = -1.0e4*g2

  J[6] = 0.0
  J[7] = 6.0e7*g2
  J[8] = 0.0

  return 0


def kinErrHandler(error_code, module, function, msg, udata):
  raise Exception(f'ERROR code {error_code} encountered in {module}::{function}: {msg}')


def solve(problem):
  print("------------------------------------------------------------\n")
  print("------------------------------------------------------------\n")
  print("Example problem from chemical kinetics solving")
  print("the first time step in a Backward Euler solution for the")
  print("following three rate equations:")
  print("    dy1/dt = -.04*y1 + 1.e4*y2*y3")
  print("    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2")
  print("    dy3/dt = 3.e2*(y2)^2")
  print("on the interval from t = 0.0 to t = 0.1, with initial")
  print("conditions: y1 = 1.0, y2 = y3 = 0.")
  print("Solution method: Newton with dense linear solver.\n")

  start = timer()

  sunctx = kin.Context().get()

  # --------------------------------------
  # Create vectors for solution and scales
  # --------------------------------------

  # Underlying data arrays for the vectors
  y = np.zeros(problem.NEQ)
  scale = np.zeros(problem.NEQ)

  # Create N_Vector objects
  sunvec_y = kin.N_VMake_Serial(y, sunctx)
  sunvec_scale = kin.N_VMake_Serial(scale, sunctx)

  # -----------------------------------------
  # Initialize and allocate memory for KINSOL
  # -----------------------------------------

  kmem = kin.KINCreate(sunctx)

  # Attach the problem object as the kinsol user_data so that
  # it is provided to the callback functions.
  kin.KINSetUserData(kmem, problem)

  #----------------------------------------------------------------------------
  # It is required to wrap Python callback functions so that they are callable
  # from C. The kin.Register<>Fn functions should be used to do this. There are
  # a few options:
  #----------------------------------------------------------------------------

  sysfn = kin.RegisterFn(funcRobertsCtypes, kin.cfunctypes.KINSysFn)

  # initialize the solver
  flag = kin.KINInit(kmem, sysfn, sunvec_y)
  if flag < 0: raise RuntimeError(f'KINInitPy returned {flag}')

  # --------------------------
  # Set KINSOL optional inputs
  # --------------------------

  # specify stopping tolerance based on residual
  fnormtol = problem.TOL
  flag = kin.KINSetFuncNormTol(kmem, fnormtol)
  if flag < 0: raise RuntimeError(f'KINSetFuncNormTol returned {flag}')

  # set a custom error handler
  flag = kin.KINSetErrHandlerFn(kmem, kin.RegisterFn(kinErrHandler, kin.cfunctypes.KINErrHandlerFn), problem)
  if flag < 0: raise RuntimeError(f'KINSetErrHandlerFn returned {flag}')

  # create and attach a linear solver
  A = kin.SUNDenseMatrix(problem.NEQ, problem.NEQ, sunctx)
  LS = kin.SUNLinSol_Dense(sunvec_y, A, sunctx)
  flag = kin.KINSetLinearSolver(kmem, LS, A)
  if flag < 0: raise RuntimeError(f'KINSetLinearSolver returned {flag}')

  # attach Jacobian function
  flag = kin.KINSetJacFn(kmem, kin.RegisterFn(jacFuncCtypes, kin.cfunctypes.KINLsJacFn))
  if flag < 0: raise RuntimeError(f'KINSetJacFn returned {flag}')

  # -------------
  # Initial guess
  # -------------

  y[0] = problem.Y10
  y[1] = problem.Y20
  y[2] = problem.Y30

  # ----------------------------
  # Call KINSOL to solve problem
  # ----------------------------

  # no scaling used
  kin.N_VConst(1.0, sunvec_scale)

  # call main solver
  flag = kin.KINSol(kmem,          # KINSOL memory block
                    sunvec_y,      # initial guess on input; solution vector
                    kin.KIN_NONE,  # global strategy choice
                    sunvec_scale,  # scaling vector for the variable cc
                    sunvec_scale)  # scaling vector for function values fval
  if flag < 0: raise RuntimeError(f'KINSol returned {flag} ({kin.KINGetReturnFlagName(flag)})')

  # ------------------------------------
  # Print solution and solver statistics
  # ------------------------------------
  flag, fnorm = kin.KINGetFuncNorm(kmem)
  if flag < 0: raise RuntimeError(f'KINGetFuncNorm returned {flag}')

  print('Computed solution (||F|| = %Lg):\n' % fnorm)
  print('y = %14.6e  %14.6e  %14.6e' % (y[0], y[1], y[2]))
  PrintFinalStats(kmem)

  # -----------
  # Free memory
  # -----------

  kin.N_VDestroy(sunvec_y)
  kin.N_VDestroy(sunvec_scale)
  kin.KINFree(kmem)

  end = timer()
  print("\nElapsed time: %g\n" % (end - start))


def PrintFinalStats(kmem):
  flag, nni = kin.KINGetNumNonlinSolvIters(kmem)
  if flag < 0: raise RuntimeError(f'KINGetNumNonlinSolvIters returned {flag}')
  flag, nli = kin.KINGetNumLinIters(kmem)
  if flag < 0: raise RuntimeError(f'KINGetNumLinIters returned {flag}')
  flag, nfe = kin.KINGetNumFuncEvals(kmem)
  if flag < 0: raise RuntimeError(f'KINGetNumFuncEvals returned {flag}')

  print('\nFinal Statistics...')
  print('nni      = %6ld      nli = %6ld      nfe     = %6ld' % (nni, nli, nfe))


if __name__ == '__main__':
  solve(Problem())