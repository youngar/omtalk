#if !defined(OMTALK_GC_BYTES_H_)
#define OMTALK_GC_BYTES_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace omtalk {

/// Get the number of bytes in n bytes.
constexpr std::size_t bytes(size_t n) { return n; }

/// Get the number of bytes in n kibibytes.
constexpr std::size_t kibibytes(std::size_t n) { return n * bytes(1024); }

/// Get the number of bytes in n mebibytes
constexpr std::size_t mebibytes(std::size_t n) { return n * kibibytes(1024); }

/// Get the number of bytes in n gibibytes.
constexpr std::size_t gibibytes(std::size_t n) { return n * mebibytes(1024); }

/// True if x is a power of two.
constexpr bool isPow2(std::size_t x) { return x && ((x & (x - 1)) == 0); }

/// The maximum safe alignment, when aligning sizes up to UNALIGNED_SIZE_MAX.
constexpr std::size_t ALIGNMENT_MAX =
    (std::numeric_limits<std::size_t>::max() >> 1) + 1;

/// The maximum safe size, when aligning up to ALIGNMENT_MAX.
constexpr std::size_t UNALIGNED_SIZE_MAX =
    (std::numeric_limits<std::size_t>::max() >> 1) + 1;

/// True if size is aligned to alignment. No safety checks.
/// alignment must be a power of two.
template <typename T>
constexpr bool alignedNoCheck(T size, std::size_t alignment) {
  return (size & (alignment - 1)) == 0;
}

/// True if size is aligned to alignment. No safety checks.
/// alignment must be a power of two.
template <typename T>
constexpr bool alignedNoCheck(T *ptr, std::size_t alignment) {
  return alignedNoCheck(std::uintptr_t(ptr), alignment);
}

/// True if size is aligned to alignment.
/// alignment must be a power of two.
template <typename T>
inline bool aligned(T size, std::size_t alignment) {
  assert(isPow2(alignment));
  return alignedNoCheck(size, alignment);
}

/// Round a size up to a multiple of alignment. No safety checks.
/// alignment must be a power of two.
template <typename T>
constexpr std::size_t alignNoCheck(T size, std::size_t alignment) {
  return (size + (alignment - 1)) & ~(alignment - 1);
}

/// Round a pointer up to a multiple of alignment. No safety checks.
/// alignment must be a power of two.
template <typename T>
constexpr std::size_t alignNoCheck(T *ptr, std::size_t alignment) {
  return alignNoCheck(std::uintptr_t(ptr), alignment);
}

/// Round a size up to a multiple of alignment.
/// alignment must be a power of two.
///
/// The maximum unaligned size is intrinsically related to the alignment.
/// As a conservative measure, users should not align sizes greater than
/// UNALIGNED_SIZE_MAX.
template <typename T, typename U>
T align(T size, U alignment) {
  assert(isPow2(alignment));
  assert(size <=
         std::numeric_limits<T>::max() - alignment + 1); // overflow check
  return alignNoCheck(size, alignment);
}

/// Round a pointer up to a multiple of alignment.
/// alignment must be a power of two.
///
/// The maximum unaligned size is intrinsically related to the alignment.
/// As a conservative measure, users should not align sizes greater than
/// UNALIGNED_SIZE_MAX.
template <typename T, typename U>
T *align(T *ptr, U alignment) {
  return static_cast<T *>(align(std::uintptr_t(ptr), alignment));
}

} // namespace omtalk

#endif // OMTALK_GC_BYTES_H_
