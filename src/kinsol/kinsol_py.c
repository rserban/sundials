
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <math.h>

#include <sundials/sundials_math.h>
#include <kinsol/kinsol_py.h>

struct KINPyUserFunctionRegistry;
static struct KINPyUserFunctionRegistry kin_pyfnregistry;

/*
 * TODO: can we autogenerate the below??
 */

struct KINPyUserFunctionRegistry
{
  KINPySysFn _KINPySysFn_;
  KINPyErrHandlerFn _KINPyErrHandlerFn_;
  KINPyInfoHandlerFn _KINPyInfoHandlerFn_;
  KINPyLsJacFn _KINPyLsJacFn_;
};

int KINPySysFn_Director(N_Vector uu, N_Vector fval, void* user_data)
{
  // TODO: If using the GPU, we need to pass back N_VGetDeviceArrayPointer instead.
  //       One way to do this would be make a compile-time decision, i.e. if CUDA
  //       is enabled this calls N_VGetDeviceArrayPointer.
  return kin_pyfnregistry._KINPySysFn_(N_VGetArrayPointer(uu), N_VGetLength(uu), N_VGetArrayPointer(fval), N_VGetLength(fval), user_data);
}

KINSysFn KINPyRegister_KINPySysFn(KINPySysFn f)
{
  kin_pyfnregistry._KINPySysFn_ = f;
  return KINPySysFn_Director;
}

void KINPyErrHandlerFn_Director(int error_code,
                                const char *module, const char *function,
                                char *msg, void *user_data)
{
  kin_pyfnregistry._KINPyErrHandlerFn_(error_code, module, function, msg, user_data);
}

KINErrHandlerFn KINPyRegister_KINPyErrHandlerFn(KINPyErrHandlerFn f)
{
  kin_pyfnregistry._KINPyErrHandlerFn_ = f;
  return KINPyErrHandlerFn_Director;
}

void KINPyInfoHandlerFn_Director(const char *module, const char *function,
                                 char *msg, void *user_data)
{
  kin_pyfnregistry._KINPyInfoHandlerFn_(module, function, msg, user_data);
}

KINInfoHandlerFn KINPyRegister_KINPyInfoHandlerFn(KINPyInfoHandlerFn f)
{
  kin_pyfnregistry._KINPyInfoHandlerFn_ = f;
  return KINPyInfoHandlerFn_Director;
}

int KINPyLsJacFn_Director(N_Vector u, N_Vector fu, SUNMatrix J,
                          void* user_data, N_Vector tmp1, N_Vector tmp2)
{
  sunindextype J_len;
  realtype* J_arr;
  SUNMatArrayView(J, &J_len, &J_arr);
  return kin_pyfnregistry._KINPyLsJacFn_(N_VGetArrayPointer(u), N_VGetLength(u),
                                         N_VGetArrayPointer(fu), J_arr, J_len, user_data,
                                         N_VGetArrayPointer(tmp1), N_VGetArrayPointer(tmp2));
}

KINLsJacFn KINPyRegister_KINPyLsJacFn(KINPyLsJacFn f)
{
  kin_pyfnregistry._KINPyLsJacFn_ = f;
  return KINPyLsJacFn_Director;
}