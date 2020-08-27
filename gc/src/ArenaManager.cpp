#include <algorithm>
#include <omtalk/ArenaAllocator.h>

void *omtalk::ArenaManager::allocate() noexcept {
  if (!cache.empty()) {
    auto arena = cache.front();
    cache.pop_front();
    return arena;
  }

  auto arena = reserve();
  if (arena != nullptr) {
    map.set(arena);
    ++arenaCount;
  }
  return arena;
}

void omtalk::ArenaManager::free(void *arena) noexcept {
  assert(arena != nullptr);
  assert(omtalk::aligned(arena, ARENA_SIZE));
  assert(this->managed(arena));

  // put arena into the cache.

  cache.push_back(arena);
  std::heap_push(cache.begin(), cache.end(), std::greater<void *>());

  // Reduce the cache size by dropping highest-addressed.

  if (arenaCount / CACHE_RATIO_DIVISOR < cache.size()) {
    auto a = cache.back();
    cache.pop_back();
    unreserve(a);
  }
}

void *omtalk::ArenaManager::reserveSlow(void *hint) noexcept {
  // Arenas are aligned to their own size, which is much larger than page
  // alignment. We map in 2*ARENA_SIZE, and discard the surplus bytes in the
  // front about back.
  assert(aligned(hint, ARENA_SIZE));
  const auto size = ARENA_SIZE * 2;

  const auto ptr = reinterpret_cast<char *>(
      mmap(hint, size, PROT_NONE, MAP_ANONYMOUS, -1, 0));
  if (ptr == MAP_FAILED) {
    return nullptr;
  }

  auto alignedPtr = omtalk::align(ptr, ARENA_SIZE);

  auto leadingBytes = arena - ptr;
  assert(omtalk::aligned(leadingBytes, omtalk::kib(4)));
  if (leadingBytes != 0)
    munmap(ptr, leadingBytes);

  auto trailingBytes = ptr + size - leadingBytes;
  assert(omtalk::aligned(trailingBytes, omtalk::kib(4)));
  if (trailingBytes != 0)
    munmap(alignedPtr + ARENA_SIZE, trailingBytes);

  hint = alignedPtr + ARENA_SIZE;
  ++arenaCount;
  map.set(alignedPtr);

  return alignedPtr;
}

void *omtalk::ArenaManager::reserveFast() noexcept {
  const auto size = 2 * ARENA_SIZE;

  const auto ptr = reinterpret_cast<char *>(
      mmap(hint, ARENA_SIZE, PROT_NONE, MAP_ANONYMOUS, -1, 0));

  if (ptr == MAP_FAILED) {
    return nullptr;
  }

  if (!aligned(ptr, ARENA_SIZE)) {
    munmap(ptr, ARENA_SIZE);
    return nullptr;
  }

  hint = ptr + ARENA_SIZE;
  ++arenaCount;
  map.set(ptr);

  return ptr;
}

void *omtalk::ArenaManager::reserve() noexcept {
  assert(aligned(hint, ARENA_SIZE));
  if (hint) {
    auto ptr = reserveFast(hint);
    if (ptr != nullptr) {
      return ptr;
    }
  }
  return reserveSlow(hint);
}

void *omtalk::ArenaManager::reserve() noexcept {
  return reserveArenaSlow(nullptr);
}

void omtalk::ArenaManager::unreserve(void *arena) {
  assert(omtalk::aligned(arena, ARENA_SIZE));
  munmap(arena, ARENA_SIZE);

  // if we're freeing an arena, it must be:
  // 1. the highest arena in the cache.
  // 2. lower than the current hint.
  hint = arena;
  --arenaCount;
  map.unset(arena);
}
