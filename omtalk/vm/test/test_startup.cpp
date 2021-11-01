#include <gtest/gtest.h>
#include <omtalk/omtalk.hpp>

TEST(Startup, startup) {
  omtalk::Process process;
  omtalk::Thread thread(process);
  omtalk::VirtualMachine vm(thread);
}