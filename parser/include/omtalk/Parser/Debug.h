#ifndef OMRTALK_PARSER_DEBUG_H_
#define OMRTALK_PARSER_DEBUG_H_

#include <iostream>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>

namespace omtalk {
namespace parser {

inline std::ostream &operator<<(std::ostream &os, const Position &pos) {
  return os << pos.line << ":" << pos.col;
}

inline std::ostream &operator<<(std::ostream &os, const Location &loc) {

  // clang-format off

  if (loc.filename == "<unknown>") {
    return os << "loc(unknown)";
  }

  // file:line:col
  if (loc.start == loc.end) {
    return os
      << "loc("
      << loc.filename
      << ':'
      << loc.start.line
      << ':'
      << loc.start.col
      << ')';
  }

  // file:line.col-col
  if (loc.start.line == loc.end.line) {
    return os
      << "loc("
      << loc.filename
      << ':'
      << loc.start.line
      << '.'
      << loc.start.col
      << '-'
      << loc.end.col
      << ')';
  }

  // file:line.col-line.col
  return os
    << "loc("
    << loc.filename
    << ":"
    << loc.start.line
    << '.'
    << loc.start.col
    << '-'
    << loc.end.line
    << '.'
    << loc.end.col
    << ')';

  // clang-format on
}

inline std::ostream &operator<<(std::ostream &os,
                                const Identifier &identifier) {
  os << "id(" << identifier.value << ", " << identifier.location << ")";
}

inline std::ostream &operator<<(std::ostream &os, const KlassDecl &klassDecl) {
  os << "class " << klassDecl.name.value;
  if (klassDecl.super) {
    os << " : " << klassDecl.super->value;
  }
  os << " {";

  if (klassDecl.fields.elements.size() != 0) {
    os << "\n  fields {";
    for (const auto &id : klassDecl.fields.elements) {
      os << " " << id.value;
    }
    os << " }\n";
  }

  os << "} " << klassDecl.location;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Module &module) {
  os << "module {";
  for (const auto &klassDecl : module.klassDecls) {
    os << "\n" << *klassDecl;
  }
  os << "\n} " << module.location;
  return os;
}

class AstPrinter {
public:
  AstPrinter(std::ostream &out, bool printLocations = false)
    : out(&out), printLocations(printLocations) {}

  AstPrinter &incNesting(std::size_t n = 1) noexcept {
    depth += n;
    return *this;
  }

  AstPrinter &decNesting(std::size_t n = 1) noexcept {
    depth -= n;
    return *this;
  }

  AstPrinter &indent() {
    for (std::size_t i = 0; i < depth; ++i) {
      *out << "  ";
    }
    return *this;
  }

  AstPrinter &printLocation(const Location& location) {
    if (printLocations) {
      *out << location;
    }
    return *this;
  }

  AstPrinter &printModule(const Module &module) {
    *out << module;
    return *this;
  }

  AstPrinter &printText(const char *str) {
    *out << str;
    return *this;
  }

private:
  bool printLocations;
  std::size_t depth;
  std::ostream *out;
};

inline void print(const Module &module, std::ostream &out, bool printLocations = false) {
  AstPrinter printer(out, printLocations);
  printer.printModule(module);
  printer.printText("\n");
}

} // namespace parser
} // namespace omtalk

#endif
