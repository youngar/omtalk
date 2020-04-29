#ifndef OMTALK_VMSTRUCTS_H_
#define OMTALK_VMSTRUCTS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct OmtalkVM {};

struct OmtalkThread {
  struct OmtalkVM *vm;
  uint8_t* pc;
  uint8_t* sp;
  uint8_t* bp;
  uint8_t* self;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // OMTALK_VMSTRUCTS_H_