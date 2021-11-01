#ifndef OM_GC_FORWARDINGMAP_H
#define OM_GC_FORWARDINGMAP_H

#include <ab/Util/Assert.h>
#include <ab/Util/Eytzinger.h>
#include <atomic>
#include <functional>

namespace om::gc {

/// Represents a forwarding from one address to another.  Has thread safe
/// guarantees to ensure that is only forwarded once. To lock the entry the To
/// field is set to 1, meaning it is invalid to set the address to s0x1.
class ForwardingEntry {
public:
  /// Create an empty forwarding to nullptr
  explicit ForwardingEntry() noexcept : to(0) {}

  /// Try to lock the to field.  Will return false if someone else has locked
  /// the entry, or the To field has been previously set.
  bool tryLock() noexcept {
    std::uintptr_t expected = 0;
    return to.compare_exchange_strong(expected, LOCKED,
                                      std::memory_order_acquire);
  }

  /// Returns true if the To entry is locked.
  bool isLocked() const noexcept { return to == LOCKED; }

  /// Get the forwarding to address. Will block if another thread is setting the
  /// address.
  void *get() const noexcept {
    // Wait if the entry is locked.
    while (to == LOCKED) {
      // spin
    }
    return reinterpret_cast<void *>(to.load());
  }

  /// Set the To entry.  This will unlock the To entry. After setting the To
  /// entry, tryLockTo() will always fail.
  void set(void *address) noexcept {
    AB_ASSERT_MSG(isLocked(), "Entries must be locked before setting");
    to.store(reinterpret_cast<std::uintptr_t>(address),
             std::memory_order_release);
  }

private:
  static constexpr std::uintptr_t LOCKED = 0x1;
  std::atomic<std::uintptr_t> to;
};

/// Maps a heap address to its forwarded address.
class ForwardingMap {
public:
  /// Construct an empty ForwardingMap
  explicit ForwardingMap() noexcept {}

  ~ForwardingMap() noexcept {}

  /// Rebuild the map by inserting all from addressess in the range [first,
  /// last].
  template <typename Iter>
  void rebuild(Iter first, Iter last) noexcept {
    map.rebuild(first, last);
  }

  /// Reset the table, creating a new table with empty forwarding information.
  template <typename Iter>
  void rebuild(Iter iter, std::size_t size) noexcept {
    map.rebuild(iter, size);
  }

  /// Clear all entries from the map and free the underlying storage.
  void clear() noexcept { map.clear(); }

  /// Return the number of entries in the map.
  std::size_t size() const noexcept { return map.size(); }

  /// Get the forwarding entry for an address.  The address must have been
  /// previously inserted.
  ForwardingEntry &at(void *address) noexcept { return map.at(address); }

  ForwardingEntry &operator[](void *address) noexcept { return map[address]; }

private:
  ab::EytzingerTree<void *, ForwardingEntry> map;
};

} // namespace om::gc

#endif