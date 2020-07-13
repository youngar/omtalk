#include "ParseCursor.h"
#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <omtalk/Parser/Debug.h>
#include <omtalk/Parser/Parser.h>
#include <sstream>

using namespace omtalk;
using namespace omtalk::parser;

namespace {

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

std::string slurp(const std::string &filename) {
  std::ifstream in(filename, std::ios::in);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void diag(const std::string& filename, const Position &position, const std::string &message) {
  std::cerr << filename << ':' << position.line << ':' << position.col << ": " << message << "\n";
}

void diag(const std::string& filename, const Position &start, const Position& end, const std::string &message) {
  // todo: emit a diagnostic that reports a range.
  diag(filename, start, message);
}

void diag(const Location &loc, const std::string &message) {
  if (loc.start == loc.end) {
    diag(loc.filename, loc.start, message);
  } else {
    diag(loc.filename, loc.start, loc.end, message);
  }
}

void diag(const ParseCursor &cursor, const std::string &message) {
  diag(cursor.loc(), message);
}

void abort(const Location &loc, const std::string &message) {
  diag(loc, "error: " + message);
  std::abort();
}

void abort(const ParseCursor &cursor, const std::string &message) {
  diag(cursor, "error: " + message);
  std::abort();
}

//===----------------------------------------------------------------------===//
// Generic Parsing
//===----------------------------------------------------------------------===//

void skipWhitespace(ParseCursor &cursor) {
  while (cursor.more() && std::isspace(*cursor)) {
    ++cursor;
  }
}

//===----------------------------------------------------------------------===//
// Omtalk Parsing
//===----------------------------------------------------------------------===//

/// Returns if the symbol is a smalltalk operator
constexpr bool isOperator(char c) {
  switch (c) {
  case '&': // and
  case '|': // or
  case '~': // not
  case '*': // star / mult
  case '@': // at
  case ',':
  case '+':
  case '-':
  case '%': // per
  case '/': // div
  case '<':
  case '>':
  case '=':
  case '\\': // mod
    return true;
  default:
    return false;
  }
}

inline void parseComment(ParseCursor &cursor) {
  assert(cursor.get() == '"');
  auto loc = cursor.loc();

  do {
    ++cursor;
    if (cursor.atEnd()) {
      abort(loc, "un-closed comment");
    } else if (*cursor == '\"') {
      ++cursor;
      break;
    } else if (*cursor == '\\') {
      ++cursor;
      if (cursor.atEnd()) {
        abort(loc, "un-closed comment on escape character");
      }
    }
  } while (true);
}

inline void parseDirective(ParseCursor &cursor) {
  assert(cursor.get() == '@');
  do {
    ++cursor;
  } while (cursor.more() && cursor.get() != '\n');
}

inline void skip(ParseCursor &cursor) {
  bool cont = true;
  while (true) {
    if (cursor.atEnd()) {
      break;
    } else if (*cursor == '\"') {
      parseComment(cursor);
    } else if (*cursor == '@') {
      parseDirective(cursor);
    } else if (std::isspace(*cursor)) {
      skipWhitespace(cursor);
    } else {
      break;
    }
  }
}

void expect(ParseCursor &cursor, char c) {
  if (*cursor != c)
    abort(cursor, "expected character" + c);
  ++cursor;
}

void expectNext(ParseCursor &cursor, char c) {
  skip(cursor);
  expect(cursor, c);
}

void expect(ParseCursor &cursor, const std::string &s) {
  for (auto x : s) {
    if (*cursor == x) {
      ++cursor;
    } else {
      abort(cursor, "expected string" + s);
    }
  }
}

void expectNext(ParseCursor &cursor, std::string s) {
  skip(cursor);
  expect(cursor, s);
}

void discardUntil(ParseCursor &cursor, char c) {
  while (*cursor != c) {
    ++cursor;
  }
}

std::string parseSymbol(ParseCursor &cursor) {
  auto start = cursor.getOffset();
  if (cursor.atEnd()) {
    abort(cursor, "Error expected symbol found EOF");
  }
  if (isOperator(*cursor)) {
    ++cursor;
  } else {
    while (cursor.more()) {
      if (*cursor == ':') {
        ++cursor;
        break;
      }
      if (!std::isalnum(*cursor)) {
        break;
      }
      ++cursor;
    }
  }
  return cursor.subStringFrom(start);
}

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

Location mkloc(Position start, const ParseCursor &cursor) {
  return {cursor.getFilename(), start, cursor.pos()};
}

Identifier parseIdentifier(ParseCursor &cursor) {
  auto start = cursor.pos();
  auto value = parseSymbol(cursor);
  return {mkloc(start, cursor), value};
}

LitInteger parseInteger(ParseCursor &cursor) {
  assert(std::isdigit(cursor.get()));
  auto start = cursor.getOffset();
  auto loc = cursor.loc();

  while (cursor.more() && std::isdigit(cursor.get())) {
    ++cursor;
  }

  return LitInteger(loc, std::stoi(cursor.subStringFrom(start)));
}

//===----------------------------------------------------------------------===//
// Class
//===----------------------------------------------------------------------===//

VarList parseVarList(ParseCursor &cursor) {
  VarList list;
  auto start = cursor.pos();

  expect(cursor, '|');

  while (true) {
    skip(cursor);

    if (cursor.atEnd()) {
      abort(cursor, "Unexpected end of variable list");
    }

    if (*cursor == '|') {
      ++cursor;
      list.location.end = cursor.pos();
      break;
    }

    if (std::isalpha(*cursor)) {
      list.elements.push_back(parseIdentifier(cursor));
    } else {
      abort(cursor, "Invalid character in variable list");
    }
  }

  list.location = mkloc(start, cursor);
  return list;
}

std::unique_ptr<KlassDecl> parseKlass(ParseCursor &cursor) {
  // class name

  auto klassDecl = std::make_unique<KlassDecl>();
  auto start = cursor.pos();

  /// ClassName

  klassDecl->name = parseIdentifier(cursor);

  // =
  
  skip(cursor);
  expectNext(cursor, '=');
  skip(cursor);

  // superklass?

  if (std::isalpha(*cursor)) {
    klassDecl->super = parseIdentifier(cursor);
    skip(cursor);
  }

  // (

  expectNext(cursor, '(');

  // Parse members

  for (bool cont = true; cont;) {
    skip(cursor);
    if (cursor.atEnd()) {
      abort(cursor, "Unexpected end of class" + klassDecl->name.value);
    } else if (*cursor == '|') {
      if (klassDecl->fields.elements.size()) {
        abort(cursor, "Class " + klassDecl->name.value + " has two instance variable lists");
      }
      klassDecl->fields = parseVarList(cursor);
    } else if (*cursor == ')') {
      break;
    } else {
      // clazz.methods().push_back(
      //     std::make_shared<ast::Method>(parse_method(cursor)));
    }
  }

  expectNext(cursor, ')');

  klassDecl->location = mkloc(start, cursor);
  return klassDecl;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

std::unique_ptr<Module> parseModule(ParseCursor &cursor) {
  auto module = std::make_unique<Module>(cursor.loc());
  auto start = cursor.pos();

  skip(cursor);
  while (cursor.more()) {
    module->klassDecls.push_back(parseKlass(cursor));
    skip(cursor);
  }

  module->location = mkloc(start, cursor);
  return module;
}

} // namespace

//===----------------------------------------------------------------------===//
// Parsing Files
//===----------------------------------------------------------------------===//

std::unique_ptr<Module> omtalk::parser::parseFile(std::string filename) {
  auto contents = slurp(filename);
  ParseCursor cursor(filename, contents);
  auto module = parseModule(cursor);
  return module;
}
