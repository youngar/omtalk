#ifndef OMTALK_COPY_H
#define OMTALK_COPY_H

#include <cassert>
#include <cstddef>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>

namespace omtalk::gc {

template <typename S>
class GlobalCollectorContext;

/// Returns whether the copy operation was a success or failure.  If the the
/// copy was a success, record the size of the copied object
class CopyResult {
public:
  /// Create a successful copy operation.
  static CopyResult success(std::size_t copySize) {
    return CopyResult(true, copySize);
  }

  /// Create a failed forwarded operation. The address is the address after the
  /// last copied object. If no object was copied, the address is the
  /// original destination address.
  static CopyResult fail() { return CopyResult(false, 0); }

  /// Returns the size of the newly copied object
  std::size_t getCopySize() const noexcept {
    assert(result);
    return copySize;
  }

  /// Returns if the operation was a success or failure.
  operator bool() { return result; }

private:
  CopyResult(bool result, std::size_t copySize)
      : result(result), copySize(copySize) {}

  bool result;
  std::size_t copySize;
};

/// Copy a target object into a memory range. The object must start at the `to`
/// pointer, but is allowed to end anywhere in the range. The object is allowed
/// to change size as it is moved. The result must indicate the amount of space
/// used for for the copy.
///
/// As an example, an object may wish to take more space after being moved if it
/// grows slots when moved, which is common in Java to grow a hash value slot,
/// or if two objects are combined into one new object, which is common in
/// javascript where objects grow slots as the program is running.
///
/// The default implementation assumes that the object does not change in size
/// or contents when it is copied.
template <typename S>
struct Copy {
  CopyResult operator()(GlobalCollectorContext<S> context, ObjectProxy<S> from,
                        std::byte *to, std::size_t size) const noexcept {
    std::size_t copySize = getCopySize<S>(from.asRef());
    if (copySize > size) {
      return CopyResult::fail();
    }
    std::memcpy(to, from.asRef().get(), copySize);
    return CopyResult::success(copySize);
  }
};

template <typename S>
CopyResult copy(GlobalCollectorContext<S> &context, ObjectProxy<S> from,
                std::byte *to, std::size_t size) noexcept {
  return Copy<S>()(context, from, to, size);
}

} // namespace omtalk::gc

#endif