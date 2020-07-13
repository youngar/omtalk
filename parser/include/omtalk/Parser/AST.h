#ifndef OMRTALK_PARSER_AST_H_
#define OMRTALK_PARSER_AST_H_

#include <cstddef>
#include <memory>
#include <omtalk/Parser/Location.h>
#include <optional>
#include <string>
#include <vector>

namespace omtalk {
namespace parser {

enum class ExprKind { LitInteger, LitFloat, LitString, Variable, Send, Array };

/// Base AST element type.
class Expr {
public:
  Expr(ExprKind k, Location l) : kind(k), location(l) {}

  virtual ~Expr() = default;

  constexpr ExprKind getKind() const noexcept { return kind; }

  constexpr const Location &loc() const noexcept { return location; }

private:
  const ExprKind kind;
  const Location location;
};

using ExprPtr = std::unique_ptr<Expr>;
using ExprPtrVec = std::vector<std::unique_ptr<Expr>>;

struct Identifier {
  Location location;
  std::string value;
};

using IdentifierVec = std::vector<Identifier>;
using IdentifierPtr = std::unique_ptr<Identifier>;
using IdentifierPtrVec = std::vector<IdentifierPtr>;

class LitInteger {
public:
  LitInteger(Location location, int value) : location(location), value(value) {}

  Location location;
  int value;
};

/// List of variables of the form: | x y z |
struct VarList {
  Location location;
  IdentifierVec elements;
};

class Method {
public:
  Location location;
  Identifier selector;
  IdentifierVec parameters;
  ExprPtrVec body;
};

using MethodVec = std::vector<Method>;

/// Top level class declaration of the form MyClass = Super ()
class KlassDecl {
public:
  KlassDecl() = default;

  KlassDecl(Location location, Identifier name)
      : location(location), name(name) {}

  Location location;
  Identifier name;
  std::optional<Identifier> super;
  VarList staticFields;
  std::vector<std::unique_ptr<Method>> staticMethods;
  VarList fields;
  MethodVec methods;
};

class Module {
public:
  Module(Location location) : location(location) {}

  Location location;
  std::vector<std::unique_ptr<KlassDecl>> klassDecls;
};

} // namespace parser
} // namespace omtalk

#endif
