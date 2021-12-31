
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
 * Can we autogenerate the below??
 */

struct KINPyUserFunctionRegistry
{
  KINPySysFn _KINPySysFn_;
};

int KINPySysFnDirector(N_Vector uu, N_Vector fval, void* user_data)
{
  return kin_pyfnregistry._KINPySysFn_(N_VGetArrayPointer(uu), N_VGetLength(uu), N_VGetArrayPointer(fval), N_VGetLength(fval), user_data);
}

KINSysFn KINPyRegisterKINPySysFn(KINPySysFn f)
{
  kin_pyfnregistry._KINPySysFn_ = (KINPySysFn) f;
  return KINPySysFnDirector;
}
