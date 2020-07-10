
#include <omtalk/vmstructs.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <omtalk/bytecodes.hpp>
#include <omtalk/interpreter.hpp>
#include <omtalk/vm/handle.hpp>
#include <omtalk/klass.hpp>

namespace omtalk {

// clang-format off
#define FOREACH_STATE_VAR(x, ...)     \
  x(__VA_ARGS__, std::uint8_t*, pc)   \
  x(__VA_ARGS__, std::uint8_t*, sp)   \
  x(__VA_ARGS__, std::uint8_t*, bp)   \
  x(__VA_ARGS__, vm::HeapPtr,   self)

#define DECLARE_STATE_VAR(thread, type, name) \
  type name = thread.name;

#define DECLARE_STATE(thread) \
  FOREACH_STATE_VAR(DECLARE_STATE_VAR, thread)

#define LOAD_STATE_VAR(thread, type, name) \
  name = thread.name;

#define LOAD_STATE(thread) \
  FOREACH_STATE_VAR(LOAD_STATE_VAR, thread)

#define SAVE_STATE_VAR(thread, type, name) \
  thread.name = name;

#define SAVE_STATE(thread) \
  FOREACH_STATE_VAR(SAVE_STATE_VAR, thread)

#define DISPATCH_INSTRUCTION(pc) \
  goto *INSTRUCTION_TABLE[load_bc(pc)]

#define DISPATCH_SEND(method_type) \
  goto *SEND_TABLE[method_type]

// clang-format on

extern "C" void omtalk_interpret(OmtalkThread &thread) {
  // clang-format off

  void *const INSTRUCTION_TABLE[] = {
    [HALT]        = &&do_halt,
    [NOP]         = &&do_nop,
    [RETURN]      = &&do_return,
    [PUSH_CONST]  = &&do_push_const,
    [PUSH_GLOBAL] = &&do_push_global
  };

  void *const SEND_TABLE[] = {
    [SEND_GENERIC]          = &&send_generic,
    [SEND_INTEGER_ADD]      = &&send_integer_add,
    [SEND_INTEGER_SUBTRACT] = &&send_integer_subtract
  };

  // clang-format on

  DECLARE_STATE(thread);
  DISPATCH_INSTRUCTION(pc);

  //
  // Bytecode Loop
  //

do_halt:
  SAVE_STATE(thread);
  return;

do_nop:
  pc += NOP_SIZE;
  DISPATCH_INSTRUCTION(pc);

do_return:
  // TODO do return
  goto do_halt;

do_push_const:
  pc += PUSH_CONST_SIZE;
  DISPATCH_INSTRUCTION(pc);

do_push_global:
  pc += PUSH_GLOBAL_SIZE;
  DISPATCH_INSTRUCTION(pc);

do_send:
  pc += SEND_SIZE;
  // goto SEND_TARGET[function.sendtarget]
  DISPATCH_INSTRUCTION(pc);

//
// Calling conventions
//

call_i2i:

return_i2i:

call_i2j:

return_i2j:

call_primitive:
  // run primitive
  DISPATCH_INSTRUCTION(pc);

return_primitive:

exit_interpreter:
  return;

  //
  // Send Targets
  //

send_generic:

send_integer_add:

send_integer_subtract:

send_primitive_unimplemented:

  return;
};

}  // namespace omtalk
