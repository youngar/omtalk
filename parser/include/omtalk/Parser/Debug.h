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

std::ostream &operator<<(std::ostream &os, const KlassDecl &klassDecl) {
  os << "class " << klassDecl.name << " {";

  os << "} " << klassDecl.location;
  return os;
}

std::ostream &operator<<(std::ostream &os, const Module &module) {
  os << "module {" << std::endl;
  for (const auto &klassDecl : module.klassDecls) {
        os << *klassDecl << std::endl;
    }
  os << "} " << module.location;
  return os;
}

} // namespace parser
} // namespace omtalk

#endif
