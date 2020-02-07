#ifndef OMTALK_LOADER_HPP_
#define OMTALK_LOADER_HPP_

#include <omtalk/ast.hpp>
#include <omtalk/parser.hpp>
#include <string>

namespace omtalk {

enum class KlassResolutionState { UNRESOLVED, IN_PROGRESS, RESOLVED };

class Resolution {
 public:
  virtual void resolve() const;
};

class KlassSymbolResolution : public Resolution {
 public:
  KlassSymbolResolution() {}

  virtual void resolve() const override {}

 private:
  // Klass *_klass;
  // std::string _name;
};

class SuperKlassResolution : public Resolution {
 public:
  SuperKlassResolution() {}

  virtual void resolve() const override {}
};

class KlassLoader {
 public:
  bool load_class(std::string class_name);
  KlassLoader(std::string class_path) : _class_path(class_path) {}

 private:
  std::string _class_path;
};

inline bool KlassLoader::load_class(std::string class_name) {
  bool result = false;

  std::string class_file = _class_path.append(class_name);
  ast::Root ast_root = parse_file(class_file);
}

}  // namespace omtalk

#endif  // OMTALK_LOADER_HPP_