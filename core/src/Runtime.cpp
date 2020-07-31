#include <omtalk/Objects.h>
#include <omtalk/Runtime.h>

namespace omtalk {

// void Universe::InitializeSystemClass(VMClass* systemClass,
// VMClass* superClass, const char* name) {
//     StdString s_name(name);

//     if (superClass != nullptr) {
//         systemClass->SetSuperClass(superClass);
//         VMClass* sysClassClass = systemClass->GetClass();
//         VMClass* superClassClass = superClass->GetClass();
//         sysClassClass->SetSuperClass(superClassClass);
//     } else {
//         VMClass* sysClassClass = systemClass->GetClass();
//         sysClassClass->SetSuperClass(load_ptr(classClass));
//     }

//     VMClass* sysClassClass = systemClass->GetClass();

//     systemClass->SetInstanceFields(NewArray(0));
//     sysClassClass->SetInstanceFields(NewArray(0));

//     systemClass->SetInstanceInvokables(NewArray(0));
//     sysClassClass->SetInstanceInvokables(NewArray(0));

//     systemClass->SetName(SymbolFor(s_name));
//     ostringstream Str;
//     Str << s_name << " class";
//     StdString classClassName(Str.str());
//     sysClassClass->SetName(SymbolFor(classClassName));

//     SetGlobal(systemClass->GetName(), systemClass);
// }

bool bootstrap(VirtualMachine &vm) { return true; }

} // namespace omtalk
