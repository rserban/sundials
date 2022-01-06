
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
};

int KINPySysFn_Director(N_Vector uu, N_Vector fval, void* user_data)
{
  return kin_pyfnregistry._KINPySysFn_(N_VGetArrayPointer(uu), N_VGetLength(uu), N_VGetArrayPointer(fval), N_VGetLength(fval), user_data);
}

KINSysFn KINPyRegister_KINPySysFn(KINPySysFn f)
{
  kin_pyfnregistry._KINPySysFn_ = (KINPySysFn) f;
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
  kin_pyfnregistry._KINPyErrHandlerFn_ = (KINPyErrHandlerFn) f;
  return KINPyErrHandlerFn_Director;
}

void KINPyInfoHandlerFn_Director(const char *module, const char *function,
                                 char *msg, void *user_data)
{
  kin_pyfnregistry._KINPyInfoHandlerFn_(module, function, msg, user_data);
}

KINInfoHandlerFn KINPyRegister_KINPyInfoHandlerFn(KINPyInfoHandlerFn f)
{
  kin_pyfnregistry._KINPyInfoHandlerFn_ = (KINPyInfoHandlerFn) f;
  return KINPyInfoHandlerFn_Director;
}
