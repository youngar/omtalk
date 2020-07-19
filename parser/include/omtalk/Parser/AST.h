#ifndef OMRTALK_PARSER_AST_H_
#define OMRTALK_PARSER_AST_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <omtalk/Parser/Location.h>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace omtalk {
namespace parser {

//===----------------------------------------------------------------------===//
// Basic Syntactic Elements
//===----------------------------------------------------------------------===//

struct Identifier {
  Identifier() = default;

  Identifier(Location location, std::string value)
      : location(std::move(location)), value(std::move(value)) {}

  Location location;
  std::string value;
};

using OptIdentifier = std::optional<Identifier>;
using IdentifierList = std::vector<Identifier>;
using IdentifierPtr = std::unique_ptr<Identifier>;
using IdentifierPtrVec = std::vector<IdentifierPtr>;

struct ParamList {
  Location location;
  IdentifierList elements;
};

using OptParamList = std::optional<ParamList>;

/// List of variables. eg: | x y z |
struct VarList {
  Location location;
  IdentifierList elements;
};

using OptVarList = std::optional<VarList>;

//===----------------------------------------------------------------------===//
// Base Expressions
//===----------------------------------------------------------------------===//

enum class ExprKind {
  // Special Reserved Words
  Nil,
  Bool,
  Self,
  Super,
  System,

  // Literals
  Integer,
  Float,
  String,
  Symbol,
  Array,

  // Expressions
  Identifier,
  Send,
  Block,
  Assignment,
  Return,
  NonlocalReturn,
};

/// Base AST element type.
struct Expr {
  virtual ~Expr() = default;

  template <typename T>
  T &cast() {
    static_assert(std::is_base_of_v<Expr, T>, "May only cast to an Expr type");
    assert(T::Kind == kind);
    return static_cast<T &>(*this);
  }

  template <typename T>
  const T &cast() const {
    static_assert(std::is_base_of_v<Expr, T>, "May only cast to an Expr type");
    assert(T::Kind == kind);
    return static_cast<const T &>(*this);
  }

  Location location;
  const ExprKind kind;

protected:
  explicit Expr(ExprKind k) : kind(k) {}

  explicit Expr(Location l, ExprKind k) : location(l), kind(k) {}
};

using ExprPtr = std::unique_ptr<Expr>;
using ExprPtrList = std::vector<ExprPtr>;

//===----------------------------------------------------------------------===//
// Special Expressions
//===----------------------------------------------------------------------===//

/// The reserved word 'nil'. TODO: Is this a literal?
struct NilExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Nil;

  NilExpr() : Expr(Kind) {}

  NilExpr(Location location) : Expr(location, Kind) {}

  virtual ~NilExpr() override = default;
};

using NilExprPtr = std::unique_ptr<NilExpr>;

inline NilExprPtr makeNilExpr(Location location) {
  return std::make_unique<NilExpr>(location);
}

/// The reserved words 'true' or 'false'. TODO: Is this a literal?
struct BoolExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Bool;

  BoolExpr() : Expr(Kind) {}

  BoolExpr(Location location, bool value)
      : Expr(location, Kind), value(value) {}

  virtual ~BoolExpr() override = default;

  bool value;
};

using BoolExprPtr = std::unique_ptr<BoolExpr>;

inline BoolExprPtr makeBoolExpr(Location location, bool value) {
  return std::make_unique<BoolExpr>(location, value);
}

/// The reserved word 'self'.
struct SelfExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Self;

  SelfExpr() : Expr(Kind) {}

  SelfExpr(Location location) : Expr(location, Kind) {}

  virtual ~SelfExpr() override = default;
};

using SelfExprPtr = std::unique_ptr<SelfExpr>;

inline SelfExprPtr makeSelfExpr(Location location) {
  return std::make_unique<SelfExpr>(location);
}

/// The reserved word 'super'.
struct SuperExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Super;

  SuperExpr() : Expr(Kind) {}

  SuperExpr(Location location) : Expr(location, Kind) {}

  virtual ~SuperExpr() override = default;
};

using SuperExprPtr = std::unique_ptr<SuperExpr>;

inline SuperExprPtr makeSuperExpr(Location location) {
  return std::make_unique<SuperExpr>(location);
}

/// The reserved word and special object 'system'.
struct SystemExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::System;

  SystemExpr() : Expr(Kind) {}

  SystemExpr(Location location) : Expr(location, Kind) {}

  virtual ~SystemExpr() override = default;
};

using SystemExprPtr = std::unique_ptr<SystemExpr>;

inline SystemExprPtr makeSystemExpr(Location location) {
  return std::make_unique<SystemExpr>(location);
}

//===----------------------------------------------------------------------===//
// Literal Expressions
//===----------------------------------------------------------------------===//

/// Integer Literal. eg: 1234
struct IntegerExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Integer;

  IntegerExpr() : Expr(Kind) {}

  IntegerExpr(Location location, std::int64_t value)
      : Expr(location, Kind), value(value) {}

  virtual ~IntegerExpr() override = default;

  std::int64_t value;
};

using IntegerExprPtr = std::unique_ptr<IntegerExpr>;

inline IntegerExprPtr makeIntegerExpr(Location location, std::int64_t value) {
  return std::make_unique<IntegerExpr>(location, value);
}

/// Float Literal. eg: 1234.567890
struct FloatExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Float;

  FloatExpr() : Expr(Kind) {}

  FloatExpr(Location location, double value)
      : Expr(location, Kind), value(value) {}

  virtual ~FloatExpr() override = default;

  double value;
};

using FloatExprPtr = std::unique_ptr<FloatExpr>;

inline FloatExprPtr makeFloatExpr(Location location, double value) {
  return std::make_unique<FloatExpr>(location, value);
}

/// String Literal, e.g. 'Hello world'
struct StringExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::String;

  StringExpr() : Expr(Kind) {}

  StringExpr(Location location, std::string value)
      : Expr(location, Kind), value(value) {}

  virtual ~StringExpr() override = default;

  std::string value;
};

using StringExprPtr = std::unique_ptr<StringExpr>;

inline StringExprPtr makeStringExpr(Location location, std::string value) {
  return std::make_unique<StringExpr>(location, std::move(value));
}

/// Symbol Literal, e.g. #hello #+
struct SymbolExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Symbol;

  SymbolExpr() : Expr(Kind) {}

  SymbolExpr(Location location, std::string value)
      : Expr(location, Kind), value(value) {}

  virtual ~SymbolExpr() override = default;

  std::string value;
};

using SymbolExprPtr = std::unique_ptr<SymbolExpr>;

inline SymbolExprPtr makeSymbolExpr(Location location, std::string value) {
  return std::make_unique<SymbolExpr>(location, std::move(value));
}

/// Array literal expression. eg: #(1 2 3 4)
struct ArrayExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Array;

  ArrayExpr() : Expr(Kind) {}

  ArrayExpr(Location location, ExprPtrList elements)
      : Expr(location, Kind), elements(std::move(elements)) {}

  virtual ~ArrayExpr() override = default;

  ExprPtrList elements;
};

using ArrayExprPtr = std::unique_ptr<ArrayExpr>;

inline ArrayExprPtr makeArrayExpr() { return std::make_unique<ArrayExpr>(); }

inline ArrayExprPtr makeArrayExpr(Location location, ExprPtrList elements) {
  return std::make_unique<ArrayExpr>(location, std::move(elements));
}

//===----------------------------------------------------------------------===//
// Expression Kinds
//===----------------------------------------------------------------------===//

/// A simple identifier expression. egc: abc.
/// Typically, a variable, or class name.
struct IdentifierExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Identifier;

  IdentifierExpr() : Expr(Kind) {}

  IdentifierExpr(Location location, std::string value)
      : Expr(location, Kind), value(std::move(value)) {
    assert(value != "self");
    assert(value != "super");
    assert(value != "true");
    assert(value != "false");
    assert(value != "nil");
    assert(value != "system");
  }

  virtual ~IdentifierExpr() override = default;

  std::string value;
};

using IdentifierExprPtr = std::unique_ptr<IdentifierExpr>;

inline IdentifierExprPtr makeIdentifierExpr(Location location,
                                            std::string value) {
  return std::make_unique<IdentifierExpr>(location, std::move(value));
}

struct SendExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Send;

  SendExpr() : Expr(Kind) {}

  SendExpr(Location location, IdentifierList selector)
      : Expr(location, Kind), selector(selector) {}

  SendExpr(Location location, IdentifierList selector, ExprPtrList parameters)
      : Expr(location, Kind), selector(selector),
        parameters(std::move(parameters)) {}

  virtual ~SendExpr() override = default;

  IdentifierList selector;
  ExprPtrList parameters;
};

using SendExprPtr = std::unique_ptr<SendExpr>;

inline SendExprPtr makeSendExpr() { return std::make_unique<SendExpr>(); }

inline SendExprPtr makeSendExpr(Location location, IdentifierList selector,
                                ExprPtrList parameters) {
  return std::make_unique<SendExpr>(location, std::move(selector),
                                    std::move(parameters));
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// A subexpression grouped by parenthesis.
struct ReturnExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Return;

  ReturnExpr() : Expr(Kind) {}

  ReturnExpr(Location location, ExprPtr value)
      : Expr(location, Kind), value(std::move(value)) {}

  virtual ~ReturnExpr() override = default;

  ExprPtr value;
};

using ReturnExprPtr = std::unique_ptr<ReturnExpr>;

inline ReturnExprPtr makeReturnExpr(Location location, ExprPtr value) {
  return std::make_unique<ReturnExpr>(location, std::move(value));
}

/// A subexpression grouped by parenthesis.
struct NonlocalReturnExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::NonlocalReturn;

  NonlocalReturnExpr() : Expr(Kind) {}

  NonlocalReturnExpr(Location location, ExprPtr value)
      : Expr(location, Kind), value(std::move(value)) {}

  virtual ~NonlocalReturnExpr() override = default;

  ExprPtr value;
};

using NonlocalReturnExprPtr = std::unique_ptr<NonlocalReturnExpr>;

inline NonlocalReturnExprPtr makeNonlocalReturnExpr(Location location,
                                                    ExprPtr value) {
  return std::make_unique<NonlocalReturnExpr>(location, std::move(value));
}

/// An assignment. eg x := y := 1234.
struct AssignmentExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Assignment;

  AssignmentExpr() : Expr(Kind) {}

  AssignmentExpr(Location location, Identifier identifier, ExprPtr &&value)
      : Expr(location, Kind), identifier(std::move(identifier)),
        value(std::move(value)) {}

  virtual ~AssignmentExpr() override = default;

  Identifier identifier;
  ExprPtr value;
};

using AssignmentExprPtr = std::unique_ptr<AssignmentExpr>;

inline AssignmentExprPtr makeAssignmentExpr() {
  return std::make_unique<AssignmentExpr>();
}

//===----------------------------------------------------------------------===//
// Blocks
//===----------------------------------------------------------------------===//

// A block literal. eg: [:x | y z | x do thing]
struct BlockExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Block;

  BlockExpr() : Expr(Kind) {}

  BlockExpr(Location location) : Expr(location, Kind) {}

  virtual ~BlockExpr() override = default;

  IdentifierList parameters;
  OptVarList locals;
  ExprPtrList body;
};

using BlockExprPtr = std::unique_ptr<BlockExpr>;

inline BlockExprPtr makeBlockExpr() { return std::make_unique<BlockExpr>(); }

inline BlockExprPtr makeBlockExpr(Location location) {
  return std::make_unique<BlockExpr>(location);
}

//===----------------------------------------------------------------------===//
// Methods
//===----------------------------------------------------------------------===//

/// Method definition. eg: myMethod: x = ( x + 1. )
class Method {
public:
  Location location;
  IdentifierList selector;
  IdentifierList parameters;
  OptVarList locals;
  ExprPtrList body;
};

using MethodPtr = std::unique_ptr<Method>;

inline MethodPtr makeMethod() { return std::make_unique<Method>(); }

using MethodPtrList = std::vector<MethodPtr>;

/// Top level class declaration. eg: MyClass = Super ()
class Klass {
public:
  Klass() = default;

  Klass(Location location, Identifier name)
      : location(location), name(std::move(name)) {}

  Location location;

  Identifier name;
  OptIdentifier super;

  OptVarList fields;
  MethodPtrList methods;

  OptVarList klassFields;
  MethodPtrList klassMethods;
};

using KlassPtr = std::unique_ptr<Klass>;

inline KlassPtr makeKlass() { return std::make_unique<Klass>(); }

inline KlassPtr makeKlass(Location location, Identifier name) {
  return std::make_unique<Klass>(location, std::move(name));
}

using KlassPtrList = std::vector<KlassPtr>;

class Module {
public:
  Module(Location location) : location(location) {}

  Location location;
  KlassPtrList klasses;
};

using ModulePtr = std::unique_ptr<Module>;

inline ModulePtr makeModule(Location location) {
  return std::make_unique<Module>(location);
}

} // namespace parser
} // namespace omtalk

#endif
