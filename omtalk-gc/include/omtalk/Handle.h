#ifndef OMTALK_GC_HANDLE_HPP_
#define OMTALK_GC_HANDLE_HPP_

#include <omtalk/Ref.h>

namespace omtalk::gc {

struct BlockEntry {
	Ref<void> value_;
};

class HandleBlock {
public:
	std::vector<Ref<void>> data;
};

class HandleScope;

class HandleBase {
public:
	template <typename T>
	Ref<T> get() const noexcept { return ref_static_cast<T>(*)}

private:
	Ref<void>* entry;
};

template <typename T>
class Handle final : private HandleBase {
public:
	Handle(HandleScope& scope, T* = nullptr) {

	}

	Ref<T> toRef() const noexcept { return ref; }

	T& operator*() const noexcept { return *ref; }

	T* operator->() const noexcept { return ref; }

private:
};

template <>
class Handle<void> final : private HandleBase {
public:
	Ref<void> toRef() const noexcept { return ref; }

private:
	Ref<void> ref;
};

} // namespace omtalk::gc

#endif // OMTALK_GC_HANDLE_HPP_
