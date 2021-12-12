#ifndef OM_AUXCONTEXTDATA_H_
#define OM_AUXCONTEXTDATA_H_

#include <om/Scheme.h>
#include <omtalk/MemoryManager.h>

template <>
class gc::AuxContextData<om::Scheme> {
public:
  AuxContextData() = default;

  gc::RootHandleScope &getRootHandleScope() noexcept { return rootHandleScope; }

  const gc::RootHandleScope &getRootHandleScope() const noexcept {
    return rootHandleScope;
  }

protected:
  gc::RootHandleScope rootHandleScope;
};

#endif // OM_AUXCONTEXTDATA_H_
