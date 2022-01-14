
#include <sunmatrix/sunmatrix_ginkgo.hpp>
#include "test_sunmatrix.h"

using namespace sundials;
using VecType = gko::matrix::Dense<sunrealtype>;

#if defined(USE_CUDA)
#include <nvector/nvector_cuda.h>
class NvectorType {
public:
  NvectorType(sunindextype len, sundials::Context& sunctx)
    : nv(N_VNew_Cuda(len, sunctx))
  {}
  operator N_Vector() { return nv; }
  const sunrealtype* get_const_values()
  {
    N_VCopyFromDevice_Cuda(nv);
    return N_VGetArrayPointer(nv);;
  }
private:
  N_Vector nv;
};
#elif defined(USE_HIP)
#include <nvector/nvector_hip.h>
class NvectorType {
public:
  NvectorType(sunindextype len, sundials::Context& sunctx)
    : nv(N_VNew_Hip(len, sunctx))
  {}
  operator N_Vector() { return nv; }
  const sunrealtype* get_const_values()
  {
    N_VCopyFromDevice_Hip(nv);
    return N_VGetArrayPointer(nv);;
  }
private:
  N_Vector nv;
};
#else
#include <nvector/nvector_serial.h>
class NvectorType {
public:
  NvectorType(sunindextype len, sundials::Context& sunctx)
    : nv(N_VNew_Serial(len, sunctx))
  {}
  operator N_Vector() { return nv; }
  const sunrealtype* get_const_values() { return N_VGetArrayPointer(nv); }
private:
  N_Vector nv;
};
#endif

template<typename MtxType>
int Test_Constructor(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec);
template<typename MtxType>
int Test_CopyConstructor(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec);
template<typename MtxType>
int Test_CppInterface(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec);

int Test_CInterfaceDense(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec);
int Test_CInterfaceCsr(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec);

int main(int argc, char* argv[])
{
  sundials::Context sunctx;

#if defined(USE_CUDA)
  auto gko_exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create(), true);
#elif defined(USE_HIP)
  auto gko_exec = gko::HipExecutor::create(0, gko::OmpExecutor::create(), true);
#else
  auto gko_exec = gko::OmpExecutor::create();
#endif

  {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> TESTING DENSE <<<<<<<<<<<<<<<<<<<<<\n";
    using MtxType = ginkgo::Matrix<gko::matrix::Dense<sunrealtype>>;
    Test_Constructor<MtxType>(sunctx, gko_exec);
    Test_CopyConstructor<MtxType>(sunctx, gko_exec);
    Test_CppInterface<MtxType>(sunctx, gko_exec);
    Test_CInterfaceDense(sunctx, gko_exec);
  }
  {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>> TESTING CSR <<<<<<<<<<<<<<<<<<<<<<\n";
    using MtxType = ginkgo::Matrix<gko::matrix::Csr<sunrealtype>>;
    Test_Constructor<MtxType>(sunctx, gko_exec);
    Test_CopyConstructor<MtxType>(sunctx, gko_exec);
    Test_CppInterface<MtxType>(sunctx, gko_exec);
    Test_CInterfaceCsr(sunctx, gko_exec);
  }
  return 0;
}

int check_vector_entries(VecType* x, sunrealtype expected)
{
  int fails = 0;
  x->get_executor()->synchronize();
  auto arr = x->get_const_values();
  for (int i = 0; i < x->get_size()[0]; ++i)
    fails += arr[i] != expected;
  return fails;
}

int check_vector_entries(NvectorType& x, sunrealtype expected)
{
  int fails = 0;
  auto arr = x.get_const_values();
  for (int i = 0; i < N_VGetLength(x); ++i)
    fails += arr[i] != expected;
  return fails;
}

template<typename MtxType>
int Test_Constructor(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec)
{
  static const char test[] = "Constructor works";
  std::cout << test;

  MtxType A{gko_exec, 2, 2, sunctx};
  MtxType B{gko_exec, 2, 2, sunctx};
  auto x = VecType::create(gko_exec, gko::dim<2>(2, 1));
  x->fill(1.0);
  auto b = VecType::create(gko_exec, gko::dim<2>(2, 1));
  b->fill(0.0);

  A.gkomtx()->read(gko::matrix_data<sunrealtype>(A.gkodim(), {1.0}));
  Matvec(A, x.get(), b.get());
  assert(check_vector_entries(b.get(), 2.0) == 0);

  B.gkomtx()->read(gko::matrix_data<sunrealtype>(B.gkodim(), {1.0}));
  ScaleAddI(1.0, B);
  Matvec(B, x.get(), b.get());
  assert(check_vector_entries(b.get(), 3.0) == 0);

  std::cout << " -- passed\n";

  return 0;
}

template<typename MtxType>
int Test_CopyConstructor(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec)
{
  static const char test[] = "Copy constructor works";
  std::cout << test;

  MtxType A{gko_exec, 2, 2, sunctx};
  MtxType B{A};
  auto x = VecType::create(gko_exec, gko::dim<2>(2, 1));
  x->fill(1.0);
  auto b = VecType::create(gko_exec, gko::dim<2>(2, 1));
  b->fill(0.0);

  A.gkomtx()->read(gko::matrix_data<sunrealtype>(A.gkodim(), {1.0}));
  Matvec(A, x.get(), b.get());
  assert(check_vector_entries(b.get(), 2.0) == 0);

  B.gkomtx()->read(gko::matrix_data<sunrealtype>(B.gkodim(), {1.0}));
  ScaleAddI(1.0, B);
  Matvec(B, x.get(), b.get());
  assert(check_vector_entries(b.get(), 3.0) == 0);

  std::cout << " -- passed\n";
  return 0;
}

template<typename MtxType>
int Test_CppInterface(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec)
{
  static const char test[] = "Test the CPP interface";
  std::cout << test;

  MtxType A{gko_exec, 2, 2, sunctx};
  MtxType B{A};
  auto x = VecType::create(gko_exec, gko::dim<2>(2, 1));
  x->fill(1.0);
  auto b = VecType::create(gko_exec, gko::dim<2>(2, 1));
  b->fill(0.0);

  Zero(A);
  ScaleAddI(1.0, A);
  ScaleAddI(1.0, A);
  Matvec(A, x.get(), b.get());
  assert(check_vector_entries(b.get(), 2.0) == 0);

  B.gkomtx()->read(gko::matrix_data<sunrealtype>(A.gkodim(), {1.0}));
  ScaleAdd(1.0, A, B);
  Matvec(A, x.get(), b.get());
  assert(check_vector_entries(b.get(), 4.0) == 0);

  Copy(A, B);
  Matvec(A, x.get(), b.get());
  assert(check_vector_entries(b.get(), 4.0) == 0);

  long int lenrw = A.workspaceSize();
  assert(lenrw == 4);

  std::cout << " -- passed\n";
  return 0;
}

int Test_CInterfaceDense(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec)
{
  static const char test[] = "Test the C dense interface";
  std::cout << test;

  SUNMatrix A = SUNMatrix_GinkgoDense(gko_exec, 2, 2, sunctx);
  SUNMatrix B = SUNMatClone(A);

  NvectorType x{2, sunctx};
  NvectorType b{2, sunctx};

  N_VConst(1.0, x);
  N_VConst(0.0, b);

  SUNMatZero(A);
  SUNMatScaleAddI(1.0, A);
  SUNMatScaleAddI(1.0, A);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 2.0) == 0);

  // SUNMatFill_GinkgoDense(1.0, B);
  SUNMatScaleAdd(1.0, A, B);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 4.0) == 0);

  SUNMatCopy(A, B);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 4.0) == 0);

  long int lenrw, leniw;
  SUNMatSpace(A, &lenrw, &leniw);
  assert(lenrw == 4);

  SUNMatDestroy(A);
  SUNMatDestroy(B);
  N_VDestroy(x);
  N_VDestroy(b);

  std::cout << " -- passed\n";
  return 0;
}

int Test_CInterfaceCsr(sundials::Context& sunctx, std::shared_ptr<gko::Executor> gko_exec)
{
  static const char test[] = "Test the C Csr interface";
  std::cout << test;

  SUNMatrix A = SUNMatrix_GinkgoCsr(gko_exec, 2, 2, sunctx);
  SUNMatrix B = SUNMatClone(A);

  NvectorType x{2, sunctx};
  NvectorType b{2, sunctx};

  N_VConst(1.0, x);
  N_VConst(0.0, b);

  SUNMatZero(A);
  SUNMatScaleAddI(1.0, A);
  SUNMatScaleAddI(1.0, A);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 2.0) == 0);

  // SUNMatFill_GinkgoCsr(1.0, B);
  SUNMatScaleAdd(1.0, A, B);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 4.0) == 0);

  SUNMatCopy(A, B);
  SUNMatMatvec(A, x, b);
  assert(check_vector_entries(b, 4.0) == 0);

  long int lenrw, leniw;
  SUNMatSpace(A, &lenrw, &leniw);
  assert(lenrw == 4);

  SUNMatDestroy(A);
  SUNMatDestroy(B);
  N_VDestroy(x);
  N_VDestroy(b);

  std::cout << " -- passed\n";
  return 0;
}

int check_matrix(SUNMatrix A, SUNMatrix B, sunrealtype tol)
{
  // int failure = 0;
  // sunindextype i = 0;
  // sunindextype Aldata = SUNMatrix_MagmaDense_LData(A);
  // sunindextype Bldata = SUNMatrix_MagmaDense_LData(B);
  // sunrealtype *Adata = (sunrealtype*) malloc(sizeof(sunrealtype)*Aldata);
  // sunrealtype *Bdata = (sunrealtype*) malloc(sizeof(sunrealtype)*Bldata);

  // /* copy data to host */
  // SUNMatrix_MagmaDense_CopyFromDevice(A, Adata);
  // SUNMatrix_MagmaDense_CopyFromDevice(B, Bdata);

  // /* check lengths */
  // if (Aldata != Bldata) {
  //   printf(">>> ERROR: check_matrix: Different data array lengths \n");
  //   return(1);
  // }

  // /* compare data */
  // for(i=0; i < Aldata; i++) {
  //   failure += SUNRCompareTol(Adata[i], Bdata[i], tol);
  // }

  // free(Adata);
  // free(Bdata);

  // if (failure > ZERO)
  //   return(1);
  // else
  //   return(0);
  return 0;
}

int check_matrix_entry(SUNMatrix A, sunrealtype val, sunrealtype tol)
{
  // int failure = 0;
  // sunindextype i = 0;
  // sunindextype Aldata = SUNMatrix_MagmaDense_LData(A);
  // sunrealtype *Adata = (sunrealtype*) malloc(sizeof(sunrealtype)*Aldata);

  // /* copy data to host */
  // SUNMatrix_MagmaDense_CopyFromDevice(A, Adata);

  // /* compare data */
  // for(i=0; i < Aldata; i++) {
  //   int check = SUNRCompareTol(Adata[i], val, tol);
  //   if (check) {
  //     printf("failed at %d\n", i);
  //     failure += check;
  //   }
  // }

  // free(Adata);

  // if (failure > ZERO)
  //   return(1);
  // else
  //   return(0);
  return 0;
}

int check_vector(N_Vector actual, N_Vector expected, sunrealtype tol)
{
  // int failure = 0;
  // sunrealtype *xdata, *ydata;
  // sunindextype xldata, yldata;
  // sunindextype i;

  // /* copy vectors to host */
  // HIP_OR_CUDA( N_VCopyFromDevice_Hip(actual);,
  //              N_VCopyFromDevice_Cuda(actual); )
  // HIP_OR_CUDA( N_VCopyFromDevice_Hip(expected);,
  //              N_VCopyFromDevice_Cuda(expected); )

  // /* get vector data */
  // xdata = N_VGetArrayPointer(actual);
  // ydata = N_VGetArrayPointer(expected);

  // /* check data lengths */
  // xldata = N_VGetLength(actual);
  // yldata = N_VGetLength(expected);


  // if (xldata != yldata) {
  //   printf(">>> ERROR: check_vector: Different data array lengths \n");
  //   return(1);
  // }

  // /* check vector data */
  // for(i=0; i < xldata; i++)
  //   failure += SUNRCompareTol(xdata[i], ydata[i], tol);

  // if (failure > ZERO) {
  //   printf("Check_vector failures:\n");
  //   for(i=0; i < xldata; i++)
  //     if (SUNRCompareTol(xdata[i], ydata[i], tol) != 0)
  //       printf("  actual[%ld] = %g != %e (err = %g)\n", (long int) i,
  //              xdata[i], ydata[i], SUNRabs(xdata[i]-ydata[i]));
  // }

  // if (failure > ZERO)
  //   return(1);
  // else
  //   return(0);
  return 0;
}

booleantype has_data(SUNMatrix A)
{
  // sunrealtype *Adata = SUNMatrix_MagmaDense_Data(A);
  // if (Adata == NULL)
  //   return SUNFALSE;
  // else
  //   return SUNTRUE;
  return SUNTRUE;
}

booleantype is_square(SUNMatrix A)
{
  // if (SUNMatrix_MagmaDense_Rows(A) == SUNMatrix_MagmaDense_Columns(A))
  //   return SUNTRUE;
  // else
  //   return SUNFALSE;
  return SUNTRUE;
}

void sync_device(SUNMatrix A)
{
}