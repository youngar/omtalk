#include "ParseCursor.h"
#include <cassert>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
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

  std::cerr << " Skipping Comment" << std::endl;

  do {
    ++cursor;
    if (cursor.atEnd()) {
      throw std::exception();
    } else if (*cursor == '\"') {
      ++cursor;
      break;
    } else if (*cursor == '\\') {
      ++cursor;
      if (cursor.atEnd()) {
        throw std::exception();
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
    throw false;
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
      // TODO how to handle errors
      throw false;
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
    throw std::exception();
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
// Class
//===----------------------------------------------------------------------===//

std::unique_ptr<ClassDecl> parseClass(ParseCursor &cursor) {

  std::cerr << " Parsing Class " << std::endl;

  Location location = cursor.loc();
  std::string name = parseSymbol(cursor);
  auto classDecl = std::make_unique<ClassDecl>(cursor.loc(), name);

  skip(cursor);
  expectNext(cursor, '=');
  skip(cursor);

  // Attempt to parse super class
  // TODO!

  expectNext(cursor, '(');

  // Parse members

  expectNext(cursor, ')');
  return classDecl;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

std::unique_ptr<Module> parseModule(ParseCursor &cursor) {
  auto module = std::make_unique<Module>(cursor.loc());

  std::cerr << " Parsing Module " << std::endl;

  skip(cursor);
  while (cursor.more()) {
    module->getClassDecls().push_back(parseClass(cursor));
    skip(cursor);
  }

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
