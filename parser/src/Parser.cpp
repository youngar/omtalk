#include "ParseCursor.h"
#include <cassert>
#include <cctype>
#include <cstring>
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
// Diagnostics
//===----------------------------------------------------------------------===//

std::string slurp(const std::string &filename) {
  std::ifstream in(filename, std::ios::in);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

Location mkloc(Position start, const ParseCursor &cursor) {
  return {cursor.getFilename(), start, cursor.pos()};
}

void diag(const std::string &filename, const Position &position,
          const std::string &message) {
  std::cerr << filename << ':' << position.line << ':' << position.col << ": "
            << message << "\n";
}

void diag(const std::string &filename, const Position &start,
          const Position &end, const std::string &message) {
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

void discardUntil(ParseCursor &cursor, char c) {
  while (*cursor != c) {
    ++cursor;
  }
}

bool match(ParseCursor &cursor, bool (*pred)(char)) {
  if (cursor.more() && pred(*cursor)) {
    ++cursor;
    return true;
  }
  return false;
}

bool match(ParseCursor &cursor, char pattern) {
  if (cursor.more() && *cursor == pattern) {
    ++cursor;
    return true;
  }
  return false;
}

bool match(ParseCursor &cursor, const char *pattern,
           std::size_t pattern_length) {
  auto c = cursor;
  for (std::size_t i = 0; i < pattern_length; ++i) {
    if (!match(c, pattern[i])) {
      return false;
    }
  }
  cursor = c;
  return true;
}

bool match(ParseCursor &cursor, const char *pattern) {
  return match(cursor, pattern, strlen(pattern));
}

bool match(ParseCursor &cursor, const std::string &pattern) {
  return match(cursor, pattern.c_str(), pattern.length());
}

/// Match any character that is not the pattern.
template <typename P>
bool match_not(ParseCursor &cursor, char pattern) {
  if (cursor.more() && *cursor != pattern) {
    ++cursor;
    return true;
  }
  return false;
}

template <typename P>
void expect(ParseCursor &cursor, const P &pattern) {
  if (!match(cursor, pattern)) {
    abort(cursor, "expected:" + pattern);
  }
}

/// match pattern N times exactly.
template <typename P>
bool n_times(ParseCursor &cursor, int count, const P &pattern) {
  auto c = cursor;
  for (std::size_t i = 0; i < count; ++i) {
    if (!match(c, pattern)) {
      return false;
    }
  }
  cursor = c;
  return true;
}

/// match C n or more times.
template <typename P>
bool n_plus(ParseCursor &cursor, int count, const P &pattern) {
  if (!n_times(cursor, count, pattern)) {
    return false;
  }
  do {
  } while (match(cursor, pattern));
  return true;
}

template <typename P>
bool one_plus(ParseCursor &cursor, const P &pattern) {
  return n_plus(cursor, 1, pattern);
}

template <typename P>
bool zero_plus(ParseCursor &cursor, const P &pattern) {
  return n_plus(cursor, 0, pattern);
}

//===----------------------------------------------------------------------===//
// Comments and Whitespace
//===----------------------------------------------------------------------===//

void skipWhitespace(ParseCursor &cursor) {
  // zero_plus(cursor, static_cast<bool(*)(char)>(std::isspace));
  zero_plus(cursor, [](char c) -> bool { return std::isspace(c); });
}

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

template <typename P>
void expectNext(ParseCursor &cursor, const P &pattern) {
  skip(cursor);
  expect(cursor, pattern);
}

template <typename P>
bool matchNext(ParseCursor &cursor, const P &pattern) {
  skip(cursor);
  return match(cursor, pattern);
}

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

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

Identifier parseIdentifier(ParseCursor &cursor) {
  auto start = cursor.pos();
  auto value = parseSymbol(cursor);
  return {mkloc(start, cursor), std::move(value)};
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
  auto klassDecl = std::make_unique<KlassDecl>();
  auto start = cursor.pos();

  // ClassName
  klassDecl->name = parseIdentifier(cursor);

  // =
  expectNext(cursor, '=');

  // superklass?
  skip(cursor);
  if (std::isalpha(*cursor)) {
    klassDecl->super = parseIdentifier(cursor);
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
        abort(cursor, "Class " + klassDecl->name.value +
                          " has two instance variable lists");
      }
      klassDecl->fields = parseVarList(cursor);
    } else if (*cursor == ')') {
      break;
    } else {
      abort(cursor, "Found somethinG???");
      // clazz.methods().push_back(
      //     std::make_shared<ast::Method>(parse_method(cursor)));
    }
  }

  // )
  expectNext(cursor, ')');

  klassDecl->location = mkloc(start, cursor);
  return klassDecl;
}

/// three or more dashes, separating class-fields and class-methods from
/// instance-fields and instance-methods.
///
bool separator(ParseCursor &cursor) { return n_plus(cursor, 3, '-'); }

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
