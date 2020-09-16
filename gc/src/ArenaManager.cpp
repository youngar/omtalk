#include <algorithm>
#include <omtalk/ArenaManager.h>

#include <cstdio>

omtalk::gc::ArenaManager::~ArenaManager() noexcept {
  for (auto arena : cache) {
    unreserve(arena);
  }

  for (auto arena : arenas) {
    unreserve(arena);
  }
}

void *omtalk::gc::ArenaManager::allocate() noexcept {
  if (!cache.empty()) {
    auto arena = cache.front();
    cache.pop_front();
    std::fprintf(stderr, "arena alloc (cache): %p\n", arena);
    arenas.insert(arena);
    return arena;
  }

  auto arena = reserve();

  if (arena != nullptr) {
    hint = alignedPtr + ARENA_SIZE;
    arenas.insert(arena);
    map.mark(arena);
  }

  return arena;
}

void omtalk::gc::ArenaManager::free(void *arena) noexcept {
  assert(arena != nullptr);
  assert(omtalk::aligned(arena, ARENA_SIZE));
  assert(this->managed(arena));

  // put arena into the cache.

  std::fprintf(stderr, "arena free: %p\n", arena);

  arenas.erase(arena);
  cache.push_back(arena);

  // Reduce the cache size by dropping highest-addressed arenas.

  auto sz = arenas.size() / CACHE_RATIO_DIVISOR;

  while (sz < cache.size()) {
    auto arena = cache.pop();
    std::fprintf(stderr, "arena dropped from cache: %p\n", arena);
    unreserve(arena);
  }
}

void *omtalk::gc::ArenaManager::reserveSlow() noexcept {
  // Arenas are aligned to their own size, which is much larger than page
  // alignment. We map in 2*ARENA_SIZE, and discard the surplus bytes in the
  // front about back.
  assert(aligned(hint, ARENA_SIZE));
  const auto size = ARENA_SIZE * 2;

  std::fprintf(stderr, "arena reserve slow: sz =%zu\n", size);

  const auto ptr = reinterpret_cast<char *>(
      mmap(hint, size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
  if (ptr == MAP_FAILED) {
    return nullptr;
  }

  auto alignedPtr = omtalk::align(ptr, ARENA_SIZE);

  auto leadingBytes = alignedPtr - ptr;
  assert(omtalk::aligned(leadingBytes, omtalk::kibibytes(4)));
  if (leadingBytes != 0)
    munmap(ptr, leadingBytes);

  auto trailingBytes = ARENA_SIZE - leadingBytes;
  assert(omtalk::aligned(trailingBytes, omtalk::kibibytes(4)));
  if (trailingBytes != 0)
    munmap(alignedPtr + ARENA_SIZE, trailingBytes);



  std::fprintf(stderr, "arena reserve slow: %p\n", alignedPtr);
  return alignedPtr;
}

void *omtalk::gc::ArenaManager::reserveFast() noexcept {
  const auto ptr = reinterpret_cast<char *>(
      mmap(hint, ARENA_SIZE, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));

  if (ptr == MAP_FAILED) {
    return nullptr;
  }

  if (!aligned(ptr, ARENA_SIZE)) {
    munmap(ptr, ARENA_SIZE);
    return nullptr;
  }

  hint = ptr + ARENA_SIZE;
  map.mark(ptr);
  std::fprintf(stderr, "arena reserve fast: %p\n", ptr);
  return ptr;
}

void *omtalk::gc::ArenaManager::reserve() noexcept {
  assert(aligned(hint, ARENA_SIZE));
  if (hint != nullptr) {
    auto ptr = reserveFast();
    if (ptr != nullptr) {
      return ptr;
    }
  }
  return reserveSlow();
}

void omtalk::gc::ArenaManager::unreserve(void *arena) noexcept {
  assert(omtalk::aligned(arena, ARENA_SIZE));
  munmap(arena, ARENA_SIZE);

  // if we're freeing an arena, it must be:
  // 1. the highest arena in the cache.
  // 2. lower than the current hint.
  hint = arena;
  map.unmark(arena);
}
