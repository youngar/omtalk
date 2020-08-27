#ifndef OMTALK_ARENAMAP_H_
#define OMTALK_ARENAMAP_H_

#include <cstddef>
#include <cstdint>
#include <omtalk/Arena.h>
#include <omtalk/Util/Atomic.h>
#include <omtalk/Util/Bytes.h>
#include <sys/mman.h>

namespace omtalk::gc {

/// This data structure tracks which arenas are reserved by the collector.
class ArenaMap {
public:
  ArenaMap() {
    chunks = reinterpret_cast<BitChunk *>(
        mmap(nullptr, NBYTES, PROT_READ | PROT_WRITE, MAP_ANONYMOUS, -1, 0));
    if (chunks == MAP_FAILED) {
      throw false;
    }
  }

  ArenaMap(const ArenaMap &) = delete;

  ArenaMap(ArenaMap &&) = default;

  ~ArenaMap() noexcept {
    if (chunks != nullptr) {
      munmap(chunks, NBYTES);
    }
  }

  /// Mark an arena pointer as being managed by the GC.
  bool mark(void *arena) noexcept {
    assert(aligned(arena, ARENA_SIZE));
    return setIndex(arenaIndex(arena));
  }

  /// Unmark an arena pointer, so that it is no longer considered managed by the
  /// GC.
  bool unmark(void *arena) noexcept {
    assert(aligned(arena, ARENA_SIZE));
    return unsetIndex(arenaIndex(arena));
  }

  /// True if the address points into memory reserved by this memory manager.
  /// This does not mean that the pointer is valid. It only indicates that
  // the memory range is reserved by the GC.
  bool managed(void *ptr) const noexcept {
    if (reinterpret_cast<void *>(MAX_ADDR) < ptr) {
      return false;
    }
    return toBool(getIndex(arenaIndex(ptr)));
  }

private:
  /// The number of bitchunks in the bitmap.
  static constexpr auto NCHUNKS = ARENA_COUNT / BITCHUNK_NBITS;

  /// The total number of bytes used to represent the bitmap.
  static constexpr auto NBYTES = NCHUNKS * sizeof(BitChunk);

  BitChunk &chunkForBit(std::size_t index) noexcept {
    return chunks[indexForBit(index)];
  }

  const BitChunk &chunkForBit(std::size_t index) const noexcept {
    return chunks[indexForBit(index)];
  }

  bool setIndex(std::size_t index) noexcept {
    if (!toBool(getIndex(index))) {
      chunkForBit(index) |= maskForBit(index);
      return true;
    }
    return false;
  }

  bool unsetIndex(std::size_t index) noexcept {
    if (toBool(getIndex(index))) {
      chunkForBit(index) &= ~maskForBit(index);
      return true;
    }
    return false;
  }

  BitChunk getIndex(std::size_t index) const noexcept {
    return chunkForBit(index) & maskForBit(index);
  }

  BitChunk *chunks;
};

} // namespace omtalk::gc

#endif // OMTALK_ARENAMAP_H_
