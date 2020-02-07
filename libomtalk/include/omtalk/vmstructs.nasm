%ifndef OMTALK_VMSTRUCTS_NASM_
%define OMTALK_VMSTRUCTS_NASM_

# OmtalkVM
struc vm
endstruc

# OmtalkThread
struc thread
    .vm:   resq 1
    .pc:   resq 1
    .sp:   resq 1
    .bp:   resq 1
    .self: resq 1
endstruc

%endif