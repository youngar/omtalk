#include <memory>
#include <om/Om/MemoryManager.h>

std::unique_ptr<om::om::RootWalker> om::om::makeRootWalker() {
  return std::make_unique<RootWalker>();
}

om::om::MemoryManager om::om::makeMemoryManager() {
  return MemoryManagerBuilder().withRootWalker(makeRootWalker()).build();
}
