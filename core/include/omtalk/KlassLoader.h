#ifndef OMTALK_KLASSLOADER_H_
#define OMTALK_KLASSLOADER_H_

#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Parser.h>

#include <optional>
#include <string>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace omtalk {

class IdentifierScope {
public:
  explicit IdentifierScope() : parent(nullptr) {}

  explicit IdentifierScope(const IdentifierScope &parent) : parent(&parent) {}

  const IdentifierScope &getParent() const { return *parent; }

  void define(const std::string &id) { ids.insert(id); }

  bool defined(const std::string &id) const {
    if (ids.count(id) != 0) {
      return true;
    }
    if (parent) {
      return parent->defined(id);
    }
    return false;
  }

private:
  const IdentifierScope *parent;
  std::unordered_set<std::string> ids;
};

class KlassLoaderEntry {};

/// The KlassLoader loads klass ASTs into memory.
/// The klass loader will load a given file, plus it's dependencies, into meory.
/// Dependencies are determined by scanning the AST.
class KlassLoader {
public:
  KlassLoader() {}

  explicit KlassLoader(std::vector<std::string> path) : path(std::move(path)) {}

  std::vector<std::string> &getPath() { return path; }

  const std::vector<std::string> &getPath() const { return path; }

  /// @group Helpers for reading classes, without actually loading them.
  /// @{

  /// find the path to the file containing a klass.
  std::optional<std::string> findKlassFile(const std::string &klassname) const {
    for (const auto &dirname : path) {
      auto filename = dirname + "/" + klassname + ".som";
      if (isLoadable(filename)) {
        return filename;
      }
    }
    return std::nullopt;
  }

  /// Given a klass name, read a klass file from the the class search path and
  /// return the file's AST. This does not load the class.
  parser::ModulePtr readKlass(const std::string &klassname) {
    auto filename = findKlassFile(klassname);
    if (filename) {
      return readFile(*filename);
    }
    return nullptr;
  }

  /// Read a specific file as an AST.
  parser::ModulePtr readFile(const std::string &filename) {
    return parser::parseFile(filename);
  }

  /// If the name is a file, load the file. If the name is a klass, load it's
  /// klass file from the class search path.
  parser::ModulePtr readFileOrKlass(const std::string &name) {
    if (isLoadable(name)) {
      return readFile(name);
    }
    return readKlass(name);
  }

  /// @}

  /// Load a file or class, validating the AST, and resolving all class
  /// references. Returns a pointer to the module containing the class
  /// if successful.
  bool loadFileOrKlass(const std::string &name) {
    auto module = readFileOrKlass(name);
    if (!module) {
      return false;
    }
    return linkModule(std::move(module));
  }

  bool loadKlass(const std::string &name) {
    auto module = readKlass(name);
    if (!module) {
      return false;
    }
    return linkModule(std::move(module));
  };

  bool loadFile(const std::string &name) {
    auto module = readFile(name);
    if (!module) {
      return false;
    }
    return linkModule(std::move(module));
  }

  const std::vector<parser::ModulePtr> &getModules() const { return modules; }

private:
  /// Resolve symbols in the module and, if successful, store the module into
  /// the table of loaded modules.
  bool linkModule(parser::ModulePtr module) {
    if (!resolveInModule(*module)) {
      return false;
    }
    modules.push_back(std::move(module));
    return true;
  }

  /// Ensure all symbols in a module have been resolved/defined.
  bool resolveInModule(const parser::Module &module) {
    for (const auto &klass : module.klasses) {
      if (!resolveInKlass(*klass)) {
        return false;
      }
    }
    return true;
  }

  /// Ensure all symbols in a klass have been resolved/defined.
  bool resolveInKlass(const parser::Klass &klass) {

    assert(!globalScope.defined(klass.name.value));
    globalScope.define(klass.name.value);

    if (klass.super) {
      if (!require(klass.super->value)) {
        return false;
      }
    }

    IdentifierScope scope(globalScope);

    if (klass.fields) {
      for (const auto &id : klass.fields->elements) {
        scope.define(id.value);
      }
    }

    for (const auto &method : klass.methods) {
      if (!resolveInMethod(scope, *method)) {
        return false;
      }
    }

    IdentifierScope klassScope(globalScope);

    if (klass.klassFields) {
      for (const auto &id : klass.klassFields->elements) {
        klassScope.define(id.value);
      }
    }

    for (const auto &method : klass.klassMethods) {
      if (!resolveInMethod(klassScope, *method)) {
        return false;
      }
    }

    return true;
  }

  /// Ensure all identifiers in a method have been resolved/defined.
  bool resolveInMethod(const IdentifierScope &parent,
                       const parser::Method &method) {
    IdentifierScope scope(parent);

    for (const auto &id : method.parameters) {
      scope.define(id.value);
    }

    if (method.locals) {
      for (const auto &id : method.locals->elements) {
        scope.define(id.value);
      }
    }

    for (const auto &expr : method.body) {
      if (!resolveInExpr(scope, *expr)) {
        return false;
      }
    }

    return true;
  }

  /// Ensure all identifiers in a method have been resolved/defined.
  bool resolveInExpr(const IdentifierScope &scope, const parser::Expr &expr) {
    switch (expr.kind) {
    case parser::ExprKind::Self:
    case parser::ExprKind::Super:
      return true;
    case parser::ExprKind::Nil:
      return require("Nil");
    case parser::ExprKind::Bool:
      return require(expr.cast<parser::BoolExpr>().value ? "True" : "False");
    case parser::ExprKind::System:
      return require("System");
    case parser::ExprKind::String:
      return require("String");
    case parser::ExprKind::Integer:
      return require("Integer");
    case parser::ExprKind::Float:
      return require("Double");
    case parser::ExprKind::Array:
      return require("Array");
    case parser::ExprKind::Symbol:
      return require("Symbol");
    case parser::ExprKind::Block:
      return resolveInBlock(scope, expr.cast<parser::BlockExpr>());
    case parser::ExprKind::Identifier:
      return require(scope, expr.cast<parser::IdentifierExpr>().value);
    case parser::ExprKind::Return:
      return resolveInExpr(scope, *expr.cast<parser::ReturnExpr>().value);
    case parser::ExprKind::NonlocalReturn:
      return resolveInExpr(scope,
                           *expr.cast<parser::NonlocalReturnExpr>().value);
    case parser::ExprKind::Send:
      return resolveInSend(scope, expr.cast<parser::SendExpr>());
    case parser::ExprKind::Assignment:
      return resolveInExpr(scope, *expr.cast<parser::AssignmentExpr>().value);
    }
  }

  bool resolveInBlock(const IdentifierScope &parent,
                      const parser::BlockExpr &block) {
    switch (block.parameters.size()) {
    case 0:
      if (!require("Block1")) {
        return false;
      }
      break;
    case 1:
      if (!require("Block2")) {
        return false;
      }
      break;
    case 2:
      if (!require("Block3")) {
        return false;
      }
      break;
    default:
      assert(0 && "Too many parameters to block");
      break;
    }

    IdentifierScope scope(parent);

    for (const auto &id : block.parameters) {
      scope.define(id.value);
    }

    if (block.locals) {
      for (const auto &id : block.locals->elements) {
        scope.define(id.value);
      }
    }

    for (const auto &expr : block.body) {
      if (!resolveInExpr(scope, *expr)) {
        return false;
      }
    }
    return true;
  }

  bool resolveInSend(const IdentifierScope &scope,
                     const parser::SendExpr &expr) {
    for (const auto &param : expr.parameters) {
      if (!resolveInExpr(scope, *param)) {
        return false;
      }
    }
    return true;
  }

  /// True if the file exists and is readable.
  bool isLoadable(const std::string &filename) const {
    return access(filename.c_str(), F_OK) != -1;
  }

  /// Require the definition of an ID in a given scope. If the ID isn't defined,
  /// load it as a klass.
  bool require(const IdentifierScope &scope, const std::string &id) {
    if (scope.defined(id)) {
      return true;
    }
    return require(id);
  }

  /// Require the definition of an ID in the global scope.
  /// If the ID isn't defined, load it as a klass.
  bool require(const std::string &id) {
    if (globalScope.defined(id)) {
      return true;
    }
    return loadKlass(id);
  }

  std::vector<std::string> path;
  IdentifierScope globalScope;
  std::vector<parser::ModulePtr> modules;
};

} // namespace omtalk

#endif