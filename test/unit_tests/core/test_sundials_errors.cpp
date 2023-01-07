/* -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2022, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------*/

#include <gtest/gtest.h>
#include <sundials/sundials.h>

class SUNErrHandlerFnTest : public ::testing::Test
{
protected:
  SUNErrHandlerFnTest() { SUNContext_Create(nullptr, &sunctx); }

  ~SUNErrHandlerFnTest() { SUNContext_Free(&sunctx); }

  SUNContext sunctx;
};

TEST_F(SUNErrHandlerFnTest, SUNLogErrHandlerFnLogsWhenCalled)
{
  testing::internal::CaptureStderr();
  std::string expected = "[ERROR][rank 0][.*][TestBody] Test log handler\n";
  SUNLogErrHandlerFn(__LINE__, __func__, __FILE__, "Test log handler", -1,
                     nullptr, sunctx);
  std::string output = testing::internal::GetCapturedStderr(); 
  ASSERT_THAT(output, testing::MatchesRegex(expected));
}

TEST_F(SUNErrHandlerFnTest, SUNAbortErrHandlerFnAbortsWhenCalled)
{
  ASSERT_DEATH(
    {
      SUNAbortErrHandlerFn(__LINE__, __func__, __FILE__, "Test abort handler",
                           -1, nullptr, sunctx);
    },
    "SUNAbortErrHandler: Calling abort now, use a different error handler to "
    "avoid program termination.\n");
}

TEST_F(SUNErrHandlerFnTest, SUNAssertErrHandlerFnAbortsWhenCalled)
{
  ASSERT_DEATH(
    {
      SUNAssertErrHandlerFn(__LINE__, __func__, __FILE__, "Test assert handler",
                            -1, nullptr, sunctx);
    },
    "SUNAssertErrHandler: assert(.*) failed... terminating\n");
}

class SUNContextErrFunctionTests : public ::testing::Test
{
protected:
  SUNContextErrFunctionTests() { SUNContext_Create(nullptr, &sunctx); }

  ~SUNContextErrFunctionTests() { SUNContext_Free(&sunctx); }

  SUNContext sunctx;
};

int firstHandler(int line, const char *func, const char *file, const char *msg,
                 SUNErrCode err_code, void *err_user_data,
                 struct SUNContext_ *sunctx)
{
  std::vector<int>* order = static_cast<std::vector<int>*>(err_user_data);
  order->push_back(0);
  return 0;
}

int secondHandler(int line, const char *func, const char *file, const char *msg,
                  SUNErrCode err_code, void *err_user_data,
                  struct SUNContext_ *sunctx)
{
  std::vector<int>* order = static_cast<std::vector<int>*>(err_user_data);
  order->push_back(1);
  return 0;
}

int thirdHandler(int line, const char *func, const char *file, const char *msg,
                 SUNErrCode err_code, void *err_user_data,
                 struct SUNContext_ *sunctx)
{
  std::vector<int>* order = static_cast<std::vector<int>*>(err_user_data);
  order->push_back(2);
  return 0;
}

TEST_F(SUNContextErrFunctionTests, SUNContextPushErrHandlerWorks)
{
  std::vector<int> order = {};
  SUNContext_ClearHandlers(sunctx);
  SUNContext_PushErrHandler(sunctx, firstHandler, static_cast<void*>(&order));
  SUNContext_PushErrHandler(sunctx, secondHandler, static_cast<void*>(&order));
  SUNContext_PushErrHandler(sunctx, thirdHandler, static_cast<void*>(&order));
  SUNHandleErr(__LINE__, __func__, __FILE__, -1, sunctx);
  ASSERT_EQ(order.size(), 3);
  ASSERT_EQ(order.at(0), 2);
  ASSERT_EQ(order.at(1), 1);
  ASSERT_EQ(order.at(2), 0);
}

TEST_F(SUNContextErrFunctionTests, SUNContextPopErrHandlerWorks)
{
  std::vector<int> order = {};
  SUNContext_ClearHandlers(sunctx);
  SUNContext_PushErrHandler(sunctx, firstHandler, static_cast<void*>(&order));
  SUNContext_PushErrHandler(sunctx, secondHandler, static_cast<void*>(&order));
  SUNContext_PushErrHandler(sunctx, thirdHandler, static_cast<void*>(&order));
  SUNContext_PopErrHandler(sunctx);
  SUNHandleErr(__LINE__, __func__, __FILE__, -1, sunctx);
  ASSERT_EQ(order.size(), 2);
  ASSERT_EQ(order.at(0), 1);
  ASSERT_EQ(order.at(1), 0);
}
