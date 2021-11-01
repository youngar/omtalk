#ifndef OM_OM_CORETYPES_H
#define OM_OM_CORETYPES_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace om::om {

// clang-format off

using i8  = std::uint8_t;
using i16 = std::uint16_t;
using i32 = std::uint32_t;
using i64 = std::uint64_t;
using f32 = float;
using f64 = double;

struct ref {
  std::uintptr_t value;
};

struct ply {
  std::uintptr_t value;
};

/// Core types are primitive types supported by the garbage collector.
enum class Type : std::uint8_t {
  /// integer types
  i8,
  i16,
  i32,
  i64,
  /// float types
  f32,
  f64,
  /// special types
  ref,
  ply,
};

template <Type T>
struct ConstTag : std::integral_constant<Type, T> {};

template <typename T>
struct TypeFor;

template <> struct TypeFor<i8>  : ConstTag<Type::i8>  {};
template <> struct TypeFor<i16> : ConstTag<Type::i16> {};
template <> struct TypeFor<i32> : ConstTag<Type::i32> {};
template <> struct TypeFor<i64> : ConstTag<Type::i64> {};
template <> struct TypeFor<f32> : ConstTag<Type::f32> {};
template <> struct TypeFor<f64> : ConstTag<Type::f64> {};
template <> struct TypeFor<ref> : ConstTag<Type::ref> {};
template <> struct TypeFor<ply> : ConstTag<Type::ply> {};

template <typename T>
using type = typename TypeFor<T>::value;

template <typename T>
struct TypeAlias {
  using type = T;
};

template <Type T>
struct CTypeFor;

template <> struct CTypeFor<Type::i8> : TypeAlias<i8> {};
template <> struct CTypeFor<Type::i16> : TypeAlias<i16> {};
template <> struct CTypeFor<Type::i32> : TypeAlias<i32> {};
template <> struct CTypeFor<Type::i64> : TypeAlias<i64> {};
template <> struct CTypeFor<Type::f32> : TypeAlias<f32> {};
template <> struct CTypeFor<Type::f64> : TypeAlias<f64> {};
template <> struct CTypeFor<Type::ref> : TypeAlias<ref> {};
template <> struct CTypeFor<Type::ply> : TypeAlias<ply> {};

template <Type T>
using ctype = CTypeFor<T>;

/// Static properties for core types.
template <Type T>
struct Prop {
  using type = ctype<T>;
  static constexpr std::size_t size = sizeof(type);
  static constexpr std::size_t alignment = alignof(type);
};

template <Type T>
constexpr std::size_t size = Prop<T>::size;

template <Type T>
constexpr std::size_t alignment = Prop<T>::alignment;

} // namespce om::om

#endif // OM_OM_CORETYPES_H
