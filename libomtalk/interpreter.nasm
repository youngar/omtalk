
%include "omtalk/vmstructs.nasm"

extern omtalk_interpret

global omtalk_interpreter

section .text

omtalk_interpreter:
    call omtalk_interpret
    ret