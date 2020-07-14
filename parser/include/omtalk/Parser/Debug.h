#ifndef OMRTALK_PARSER_DEBUG_H_
#define OMRTALK_PARSER_DEBUG_H_

#include <iostream>
#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>

namespace omtalk {
namespace parser {

class AstPrinter {
public:
  AstPrinter(std::ostream &out) : depth(0), out(&out) {}

  void incNesting(std::size_t n = 1) noexcept { depth += n; }

  void decNesting(std::size_t n = 1) noexcept { depth -= n; }

  void indent() {
    for (std::size_t i = 0; i < depth; ++i) {
      *this << "  ";
    }
  }

  template <typename T>
  friend AstPrinter &operator<<(AstPrinter &printer, const T &value) {
    printer.fresh = false;
    *printer.out << value;
    return printer;
  }

  /// Begin a record
  void enter() {
    freshen();
    indent();
    *this << "{";
    incNesting();
  }

  /// Begin a labeled record
  void enter(const std::string &label) {
    *this << label << " {";
    incNesting();
  }

  /// Leave a record.
  void leave() {
    decNesting();
    newln();
    *this << "}";
  }

  template <typename T>
  void field(const std::string &label, const T &value) {
    freshen();
    indent();
    *this << label << ": ";
    print(*this, value);
  }

  void field(const std::string &label, const std::string &value) {
    freshen();
    indent();
    *this << label << ": \"" << value << "\"";
  }

  void freshen() {
    if (!fresh) {
      *out << "\n";
      fresh = true;
    }
  }

  void newln() {
    freshen();
    indent();
  }

  void start() {
    fresh = true;
  }

private:
  int depth;
  std::ostream *out;
  bool fresh = true;
};

inline void print(AstPrinter &p, char c) { p << c; }

inline void print(AstPrinter &p, const char *str) { p << str; }

inline void print(AstPrinter &p, int i) { p << i; }

inline void print(AstPrinter &p, unsigned i) { p << i; }

inline void print(AstPrinter &p, double i) { p << i; }

inline void print(AstPrinter &p, const std::string &str) { p << str; }

template <typename T>
inline void print(AstPrinter &p, const std::unique_ptr<T> &ptr) {
  if (ptr) {
    print(p, *ptr);
  } else {
    print(p, "<nullptr>");
  }
}

template <typename T>
inline void print(AstPrinter &p, const std::optional<T> &opt) {
  if (opt) {
    print(p, *opt);
  } else {
    print(p, "<nullopt>");
  }
}

template <typename T>
inline void print(AstPrinter &p, const std::vector<T> &vec) {
  if (vec.size() == 0) {
    print(p, "[]");
    return;
  }
  print(p, "[");
  p.incNesting();
  for (const auto &e : vec) {
    p.newln();
    p.start();
    print(p, e);
  }
  p.decNesting();
  p.newln();
  print(p, "]");
}

inline void print(AstPrinter &p, const Location &loc) {
  print(p, "(");
  print(p, loc.filename);
  print(p, ":");
  print(p, loc.start.line);
  print(p, ":");
  print(p, loc.start.col);
  if (loc.start.line == loc.end.line) {
    if (loc.start.col != loc.end.col) {
      print(p, "-");
      print(p, loc.end.col);
    }
  } else {
    print(p, "-");
    print(p, loc.end.line);
    print(p, ":");
    print(p, loc.end.col);
  }
  print(p, ")");
}

inline void print(AstPrinter &p, const Identifier &identifier) {
  p.enter("Identifier");
  p.field("location", identifier.location);
  p.field("name", identifier.value);
  p.leave();
}

inline void print(AstPrinter &p, const VarList &list) {
  p.enter("VarList");
  p.field("location", list.location);
  p.field("elements", list.elements);
  p.leave();
}

inline void print(AstPrinter &p, const LitIntegerExpr &expr) {
  p.enter("LitIntegerExpr");
  p.field("location", expr.location);
  p.field("value", expr.value);
  p.leave();
}

inline void print(AstPrinter &p, const LitFloatExpr &expr) {
  p.enter("LitFloatExpr");
  p.field("location", expr.location);
  p.field("value", expr.value);
  p.leave();
}

inline void print(AstPrinter &p, const LitStringExpr &expr) {
  p.enter("LitStringExpr");
  p.field("location", expr.location);
  p.field("value", expr.value);
  p.leave();
}

inline void print(AstPrinter &p, const LitSymbolExpr &expr) {
  p.enter("LitSymbolExpr");
  p.field("location", expr.location);
  p.field("value", expr.value);
  p.leave();
}

inline void print(AstPrinter &p, const LitArrayExpr &expr) {
  p.enter("LitSymbolExpr");
  p.field("location", expr.location);
  p.field("elements", expr.elements);
  p.leave();
}

inline void print(AstPrinter &p, const IdentifierExpr &expr) {
  p.enter("LitSymbolExpr");
  p.field("location", expr.location);
  p.field("name", expr.name);
  p.leave();
}

inline void print(AstPrinter &p, const SendExpr &expr) {
  p.enter("SendExpr");
  p.field("location", expr.location);
  p.field("selector", expr.selector);
  p.field("parameters", expr.parameters);
  p.leave();
}

inline void print(AstPrinter &p, const BlockExpr &expr) {
  p.enter("BlockExpr");
  p.field("location", expr.location);
  p.field("parameters", expr.parameters);
  p.field("locals", expr.locals);
  p.field("body", expr.body);
  p.leave();
}

inline void print(AstPrinter &p, const AssignmentExpr &expr) {
  p.enter("AssignmentExpr");
  p.field("location", expr.location);
  p.field("identifiers", expr.identifiers);
  p.field("value", expr.value);
  p.leave();
}

inline void print(AstPrinter &p, const ReturnExpr &expr) {
  p.enter("ReturnExpr");
  p.field("location", expr.location);
  p.field("value", expr.value);
  p.leave();
}


inline void print(AstPrinter &p, const Expr &expr) {
  switch (expr.kind) {
  case ExprKind::LitIntegerExpr:
    print(p, expr.cast<LitIntegerExpr>());
    break;
  case ExprKind::LitFloatExpr:
    print(p, expr.cast<LitFloatExpr>());
    break;
  case ExprKind::LitStringExpr:
    print(p, expr.cast<LitStringExpr>());
    break;
  case ExprKind::LitSymbolExpr:
    print(p, expr.cast<LitSymbolExpr>());
    break;
  case ExprKind::LitArrayExpr:
    print(p, expr.cast<LitArrayExpr>());
    break;
  case ExprKind::Identifier:
    print(p, expr.cast<IdentifierExpr>());
    break;
  case ExprKind::Send:
    print(p, expr.cast<SendExpr>());
    break;
  case ExprKind::Block:
    print(p, expr.cast<BlockExpr>());
    break;
  case ExprKind::Assignment:
    print(p, expr.cast<AssignmentExpr>());
    break;
  case ExprKind::Return:
    print(p, expr.cast<ReturnExpr>());
    break;

  default:
    p.enter("expression");
    p.field("location", expr.location);
    p.field("kind", static_cast<unsigned>(expr.kind));
    p.leave();
    break;
  }
}

inline void print(AstPrinter &p, const Method &method) {
  p.enter("Method");
  p.field("location", method.location);
  p.field("selector", method.selector);
  p.field("parameters", method.parameters);
  p.field("locals", method.locals);
  p.field("body", method.body);
  p.leave();
}

inline void print(AstPrinter &p, const Klass &klass) {
  p.enter("Klass");
  p.field("location", klass.location);
  p.field("name", klass.name);
  p.field("super", klass.super);
  p.field("fields", klass.fields);
  p.field("methods", klass.methods);
  p.field("klassFields", klass.klassFields);
  p.field("klassMethods", klass.klassMethods);
  p.leave();
}

inline void print(AstPrinter &p, const Module &module) {
  p.enter("Module");
  p.field("location", module.location);
  p.field("klasses", module.klasses);
  p.leave();
}

inline void print(std::ostream &out, const Module &module) {
  AstPrinter p(out);
  print(p, module);
  print(p, "\n");
}

} // namespace parser
} // namespace omtalk

#endif
