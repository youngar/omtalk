#ifndef OMRTALK_PARSER_DEBUG_H_
#define OMRTALK_PARSER_DEBUG_H_

#include <iostream>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>

namespace omtalk {
namespace parser {

std::ostream &operator<<(std::ostream &os, const Position &pos) {
  return os << pos.line << ":" << pos.col;
}

std::ostream &operator<<(std::ostream &os, const Location &loc) {
  if (loc.filename == "<unknown>") {
    return os << "loc(unknown)";
  }

  if (loc.start == loc.end) {
    return os << "loc(" << loc.filename << ":" << loc.start << ")";
  }

  os << "loc(" << loc.filename << ":" << loc.start << "-" << loc.end << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ClassDecl &classDecl) {
  os << "class " << classDecl.getName() << " {";

  os << "} " << classDecl.loc();
  return os;
}

std::ostream &operator<<(std::ostream &os, const Module &module) {
  os << "module {" << std::endl;
  for (const auto &classDecl : module.getClassDecls()) {
        os << *classDecl << std::endl;
    }
  os << "} " << module.loc();
  return os;
}

} // namespace parser
} // namespace omtalk

#endif