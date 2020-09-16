#ifndef OMTALK_FORWARDINGMAP_H
#define OMTALK_FORWARDINGMAP_H

#include <atomic>
#include <functional>
#include <omtalk/Util/Assert.h>
#include <omtalk/Util/Eytzinger.h>

namespace omtalk::gc {

/// Represents a forwarding from one address to another.  Has thread safe
/// guarantees to ensure that is only forwarded once. To lock the entry the To
/// field is set to 1, meaning it is invalid to set the address to s0x1.
class ForwardingEntry {
public:
  /// Create an empty forwarding to nullptr
  explicit ForwardingEntry() : to(0) {}

  /// Try to lock the to field.  Will return false if someone else has locked
  /// the entry, or the To field has been previously set.
  bool tryLock() {
    std::uintptr_t expected = 0;
    return to.compare_exchange_strong(expected, LOCKED);
  }

  /// Returns true if the To entry is locked.
  bool isLocked() { return to == LOCKED; }

  /// Get the forwarding to address. Will block if another thread is setting the
  /// address.
  void *get() {
    // Wait if the entry is locked.
    while (to == LOCKED) {
      // spin
    }
    return reinterpret_cast<void *>(to.load());
  }

  /// Set the To entry.  This will unlock the To entry. After setting the To
  /// entry, tryLockTo() will always fail.
  void set(void *address) {
    OMTALK_ASSERT_MSG(isLocked(), "Entries must be locked before setting");
    to.store(reinterpret_cast<std::uintptr_t>(address));
  }

private:
  static constexpr std::uintptr_t LOCKED = 0x1;
  std::atomic<std::uintptr_t> to;
};

/// Maps a heap address to its forwarded address.
class ForwardingMap {
public:
  /// Construct an empty ForwardingMap
  explicit ForwardingMap() {}

  ~ForwardingMap() noexcept {}

  /// Rebuild the map by inserting all from addressess in the range [first,
  /// last].
  template <typename Iter>
  void rebuild(Iter first, Iter last) noexcept {
    map.rebuild(first, last);
  }

  /// Reset the table, creating a new table with empty forwarding information.
  template <typename Iter>
  void rebuild(Iter iter, std::size_t size) {
    map.rebuild(iter, size);
  }

  /// Clear all entries from the map and free the underlying storage.
  void clear() noexcept { map.clear(); }

  /// Return the number of entries in the map.
  std::size_t size() noexcept { return map.size(); }

  /// Get the forwarding entry for an address.  The address must have been
  /// previously inserted.
  ForwardingEntry &at(void *address) noexcept { return map.at(address); }

  ForwardingEntry &operator[](void *address) { return map[address]; }

private:
  EytzingerTree<void *, ForwardingEntry> map;
};

} // namespace omtalk::gc

#endif

#if 0

// .. pre-insert in the map

// .. later

/// Get a forwarded object, or forward the object
/// Record a forwarding from one address to another address.  If a forwarding
/// for an address already exists, return the previous forwarding address.
template<typename S>
static void *forwardOrGet(GlobalCollectorContext<S> cx,
                          void *address() {
    Region region = Region::get(address);
    ForwardingEntry &f = region.getForwardingTable()[address];
    void *newAddress = f.getTo();
    if (newAddress == nullptr) {
        if (f.tryLock()) {
            newAddress = allocate();
            // copy here!
            f.setTo(newAddress);
        } else {
            // Someone else is copying the entry.  Wait for them to finish.
            newAddress = f.getTo();
        }
    }
    return newAddress;
};

void rebuild(const Region &region) { auto n = region.getLiveObjectCount(); }


void *forward(Context &context, void *target) {
  auto *region = Region::get(target);
  auto &table = region->getForwardingTable();

  auto &entry = table.get(target);
  if (entry.getTo() != nullptr) {
  }
}
#endif
