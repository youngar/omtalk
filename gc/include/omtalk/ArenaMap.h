#ifndef OMTALK_ARENAMAP_H_
#define OMTALK_ARENAMAP_H_

#include <cstddef>
#include <cstdint>
#include <omtalk/Arena.h>
#include <omtalk/Util/Atomic.h>
#include <omtalk/Util/BitChunk.h>
#include <omtalk/Util/Bytes.h>
#include <sys/mman.h>

namespace omtalk::gc {

/// A massive bitmap indicating which addresses are reserved by
/// the GC. Used to filter out off-heap pointers in the GC.
class ArenaMap {
public:
  ArenaMap() {
    chunks = reinterpret_cast<BitChunk *>(
        mmap(nullptr, NBYTES, PROT_READ | PROT_WRITE,
             MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
    assert(chunks != nullptr);
    assert(chunks != MAP_FAILED);
  }

  ArenaMap(const ArenaMap &) = delete;

  ArenaMap(ArenaMap &&other) {
    chunks = other.chunks;
    other.chunks = nullptr;
  }

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
    return chunks[index / BITCHUNK_NBITS];
  }

  const BitChunk &chunkForBit(std::size_t index) const noexcept {
    return chunks[index / BITCHUNK_NBITS];
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
