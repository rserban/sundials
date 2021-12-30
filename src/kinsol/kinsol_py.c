
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <math.h>

#include <sundials/sundials_math.h>
#include <kinsol/kinsol_py.h>
#include "kinsol_impl.h"

static KINPyUserFunctionRegistry kin_pyfnregistry;

int KINPySysFnDirector(N_Vector uu, N_Vector fval, void* user_data)
{
  return kin_pyfnregistry._KINPySysFn_(N_VGetArrayPointer(uu), N_VGetLength(uu), N_VGetArrayPointer(fval), N_VGetLength(fval));
}

KINPyCallbackFn KINPyRegisterFn(KINPyCallbackFn f, const char* name)
{
  if (strcmp(name, "KINPySysFn") == 0) {
    kin_pyfnregistry._KINPySysFn_ = (KINPySysFn) f;
    return ((KINPyCallbackFn) KINPySysFnDirector);
  }
}
