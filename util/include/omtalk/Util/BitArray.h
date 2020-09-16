#ifndef OMTALK_GC_BITARRAY_HPP_
#define OMTALK_GC_BITARRAY_HPP_

#include <array>
#include <cstdint>
#include <omtalk/Util/BitChunk.h>
#include <omtalk/Util/Bytes.h>

namespace omtalk {

template <std::size_t N>
class BitArray {
public:
  static_assert((N % BITCHUNK_NBITS) == 0,
                "BitArray size must be a multiple of BITCHUNK_NBITS.");

  static constexpr std::size_t NCHUNKS = N / BITCHUNK_NBITS;

  BitArray() = default;

  bool get(std::size_t index) const noexcept {
    return BitChunk(0) != (chunkForBit(index) & maskForBit(index));
  }

  bool set(std::size_t index) noexcept {
    if (!get(index)) {
      chunkForBit(index) |= maskForBit(index);
      return true;
    }
    return false;
  }

  bool unset(std::size_t index) noexcept {
    if (get(index)) {
      chunkForBit(index) &= ~maskForBit(index);
      return true;
    }
    return false;
  }

  std::size_t size() const noexcept { return chunks.size() * BITCHUNK_NBITS; }

  void clear() noexcept { chunks.fill(BitChunk(0)); }

  using Iterator = typename BitChunkArray<NCHUNKS>::iterator;

  using ConstIterator = typename BitChunkArray<NCHUNKS>::const_iterator;

  Iterator begin() { return chunks.begin(); }

  Iterator end() { return chunks.end(); }

  ConstIterator cbegin() const { return chunks.cbegin(); }

  ConstIterator cend() const { return chunks.cend(); }

  std::size_t count() const noexcept {
    std::size_t sum = 0;
    for (auto chunk : chunks) {
      sum += count(chunk);
    }
    return sum;
  }

  bool all() const noexcept {
    for (auto chunk : chunks) {
      if (!all(chunk)) {
        return false;
      }
    }
    return true;
  }

  bool any() const noexcept {
    for (auto chunk : chunks) {
      if (any(chunk)) {
        return true;
      }
    }
    return false;
  }

private:
  static constexpr std::size_t indexForBit(std::size_t index) {
    return index / BITCHUNK_NBITS;
  }

  static constexpr std::size_t shiftForBit(std::size_t index) {
    return index % BITCHUNK_NBITS;
  }

  static constexpr BitChunk maskForBit(std::size_t index) {
    return BitChunk(1) << shiftForBit(index);
  }

  BitChunk &chunkForBit(std::size_t index) noexcept {
    return chunks.at(indexForBit(index));
  }

  const BitChunk &chunkForBit(std::size_t index) const noexcept {
    return chunks.at(indexForBit(index));
  }

  BitChunkArray<NCHUNKS> chunks;
};

class BitArray32 {
public:
  BitArray32() : data(0) {}

  BitArray32(const BitArray32 &other) : data(other.data) {}

  BitArray32(std::uint32_t data) : data(data) {}

  bool all() const noexcept { return data == UINT32_MAX; }

  bool none() const noexcept { return data == 0; }

  bool any() const noexcept { return data != 0; }

  bool test(std::size_t index) const noexcept {
    assert(index < 32);
    auto mask = std::uint32_t(1) << index;
    return (data & mask) != 0;
  }

  /// Set bit to 1.
  bool set(std::size_t index) noexcept {
    assert(index < 32);
    auto mask = std::uint32_t(1) << index;
    if (data & mask) {
      return false;
    }
    data |= mask;
    return true;
  }

  /// Set bit to zero.
  bool reset(std::size_t index) noexcept {
    assert(index < 32);
    auto mask = std::uint32_t(1) << index;
    if ((data & mask) == 0) {
      return false;
    }
    data &= ~mask;
    return false;
  }

  std::size_t size() const noexcept { return 32; }

  std::size_t count() const noexcept { return __builtin_popcount(data); }

private:
  std::uint32_t data;
};

} // namespace omtalk

#endif // OMTALK_GC_BITARRAY_HPP_
