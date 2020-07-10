#include <omtalk/omtalk.hpp>

using namespace omtalk;

// TEST(Env, ResourceDirAccurate) {
//   const char* dir = getenv("OMTALK_RESOURCEDIR");
//   if (dir == nullptr) {
//     dir = OMTALK_RESOURCEDIR;
//   }

//   EXPECT_STREQ(test::env->resourcedir().c_str(), dir);

//   EXPECT_STREQ((test::env->resourcedir() + "/hello").c_str(),
//                (std::string(dir) + "/hello").c_str());

//   EXPECT_STREQ(test::env->resource("hello").c_str(),
//                (std::string(dir) + "/hello").c_str());
// }

// TEST(Env, LoadFromResourceDirectory) {
//   std::ifstream t(test::env->resourcedir() + "/test.txt");
//   std::string str((std::istreambuf_iterator<char>(t)),
//                   std::istreambuf_iterator<char>());

//   EXPECT_STREQ(str.c_str(), "testing the tester");
// }