#ifndef OMRTALK_PARSER_AST_H_
#define OMRTALK_PARSER_AST_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <omtalk/Parser/Location.h>
#include <optional>
#include <string>
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
  LitIntegerExpr,
  LitFloatExpr,
  LitStringExpr,
  LitSymbolExpr,
  LitArrayExpr,
  Identifier,
  Send,
  Block,
  Assignment,
  Return
};

/// Base AST element type.
struct Expr {
  virtual ~Expr() = default;

  Location location;
  const ExprKind kind;

  template <typename T>
  T &cast() {
    assert(T::Kind == kind);
    return static_cast<T &>(*this);
  }

  template <typename T>
  const T &cast() const {
    assert(T::Kind == kind);
    return static_cast<const T &>(*this);
  }

protected:
  explicit Expr(ExprKind k) : kind(k) {}

  explicit Expr(Location l, ExprKind k) : location(l), kind(k) {}
};

using ExprPtr = std::unique_ptr<Expr>;
using ExprPtrList = std::vector<ExprPtr>;

//===----------------------------------------------------------------------===//
// Literal Expressions
//===----------------------------------------------------------------------===//

/// Integer Literal. eg: 1234
struct LitIntegerExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::LitIntegerExpr;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  LitIntegerExpr() : Expr(Kind) {}

  LitIntegerExpr(Location location, int value)
      : Expr(location, Kind), value(value) {}

  virtual ~LitIntegerExpr() override = default;

  int value;
};

using LitIntegerExprPtr = std::unique_ptr<LitIntegerExpr>;

/// Float Literal. eg: 1234.567890
struct LitFloatExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::LitFloatExpr;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  LitFloatExpr() : Expr(ExprKind::LitFloatExpr) {}

  LitFloatExpr(Location location, double value)
      : Expr(location, Kind), value(value) {}

  virtual ~LitFloatExpr() override = default;

  double value;
};

using LitFloatExprPtr = std::unique_ptr<LitFloatExpr>;

/// String Literal, e.g. 'Hello world'
struct LitStringExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::LitStringExpr;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  LitStringExpr() : Expr(Kind) {}

  LitStringExpr(Location location, std::string value)
      : Expr(location, Kind), value(value) {}

  virtual ~LitStringExpr() override = default;

  std::string value;
};

using LitStringExprPtr = std::unique_ptr<LitStringExpr>;

/// Symbol Literal, e.g. #hello #+
struct LitSymbolExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::LitSymbolExpr;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  LitSymbolExpr() : Expr(Kind) {}

  LitSymbolExpr(Location location, std::string value)
      : Expr(location, Kind), value(value) {}

  virtual ~LitSymbolExpr() override = default;

  std::string value;
};

using LitSymbolExprPtr = std::unique_ptr<LitSymbolExpr>;

/// Array literal expression. eg: #(1 2 3 4)
struct LitArrayExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::LitArrayExpr;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  LitArrayExpr() : Expr(Kind) {}

  LitArrayExpr(Location location, ExprPtrList elements)
      : Expr(location, Kind), elements(std::move(elements)) {}

  virtual ~LitArrayExpr() override = default;

  ExprPtrList elements;
};

using LitArrayExprPtr = std::unique_ptr<LitArrayExpr>;

//===----------------------------------------------------------------------===//
// Expression Kinds
//===----------------------------------------------------------------------===//

/// A simple identifier expression. egc: abc.
/// Typically, a variable, or class name.
struct IdentifierExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Identifier;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  IdentifierExpr() : Expr(Kind) {}

  IdentifierExpr(Location location, std::string name)
      : Expr(location, Kind), name(std::move(name)) {}

  virtual ~IdentifierExpr() override = default;

  std::string name;
};

using IdentifierExprPtr = std::unique_ptr<IdentifierExpr>;

/// An assignment. eg x := y := 1234.
struct AssignmentExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Assignment;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  AssignmentExpr() : Expr(Kind) {}

  AssignmentExpr(Location location, IdentifierList identifiers, ExprPtr value)
      : Expr(location, Kind), identifiers(std::move(identifiers)),
        value(std::move(value)) {}

  virtual ~AssignmentExpr() override = default;

  IdentifierList identifiers;
  ExprPtr value;
};

using AssignmentExprPtr = std::unique_ptr<AssignmentExpr>;

/// A subexpression grouped by parenthesis.
struct ReturnExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Return;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  ReturnExpr() : Expr(Kind) {}

  ReturnExpr(Location location, ExprPtr value)
      : Expr(location, Kind), value(std::move(value)) {}

  virtual ~ReturnExpr() override = default;

  ExprPtr value;
};

using ReturnExprPtr = std::unique_ptr<ReturnExpr>;

struct SendExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Send;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

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

//===----------------------------------------------------------------------===//
// Blocks
//===----------------------------------------------------------------------===//

// A block literal. eg: [:x | y z | x do thing]
struct BlockExpr final : public Expr {
  static constexpr ExprKind Kind = ExprKind::Block;

  static constexpr bool kindof(const Expr &expr) { return expr.kind == Kind; }

  BlockExpr() : Expr(Kind) {}

  BlockExpr(Location location) : Expr(location, Kind) {}

  virtual ~BlockExpr() override = default;

  IdentifierList parameters;
  OptVarList locals;
  ExprPtrList body;
};

using BlockExprPtr = std::unique_ptr<BlockExpr>;

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

using MethodPtrList = std::vector<MethodPtr>;

/// Top level class declaration. eg: MyClass = Super ()
class Klass {
public:
  Klass() = default;

  Klass(Location location, Identifier name)
      : location(location), name(name) {}

  Location location;

  Identifier name;
  OptIdentifier super;

  OptVarList fields;
  MethodPtrList methods;

  OptVarList klassFields;
  MethodPtrList klassMethods;
};

using KlassPtr = std::unique_ptr<Klass>;
using KlassPtrList = std::vector<KlassPtr>;

class Module {
public:
  Module(Location location) : location(location) {}

  Location location;
  KlassPtrList klassDecls;
};

} // namespace parser
} // namespace omtalk

#endif
