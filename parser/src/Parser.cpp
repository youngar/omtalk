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
#include <string>

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

Location mkloc(const ParseCursor &start, const ParseCursor &cursor) {
  return mkloc(start.pos(), cursor);
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
// Predicates
//===----------------------------------------------------------------------===//

bool isAlpha(char c) { return std::isalpha(static_cast<unsigned char>(c)); }

bool isAlnum(char c) { return std::isalnum(static_cast<unsigned char>(c)); }

bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)); }

bool isSpace(char c) { return std::isspace(static_cast<unsigned char>(c)); }

/// Returns if the symbol is a smalltalk operator
constexpr bool isOperator(char c) {
  switch (c) {
  case '&':  // and
  case '|':  // or
  case '~':  // not
  case '*':  // star / mult
  case '@':  // at
  case ',':  // comma
  case '+':  // plus
  case '-':  // minus
  case '%':  // per
  case '/':  // div
  case '<':  // less
  case '>':  // more
  case '=':  // equal
  case '\\': // mod
    return true;
  default:
    return false;
  }
}

// TODO are we parsing Binary vs Unary operators correctly?
// Symbol singleOpSyms[] = { Not, And, Or, Star, Div, Mod, Plus, Equal, More,
// Less,
//         Comma, At, Per, NONE };

// Symbol binaryOpSyms[] = { Or, Comma, Minus, Equal, Not, And, Or, Star, Div,
// Mod,
//         Plus, Equal, More, Less, Comma, At, Per, NONE };

// binarySelector:
//     Or | Comma | Minus | Equal | Not | And | Star | Div | Mod | Plus | More |
//     Less | At | Per | OperatorSequence;

constexpr bool isUnaryOperator(char c) {
  switch (c) {
  case '!': // not
    return true;
  default:
    return false;
  }
}

constexpr bool isBinaryOperator(char c) {
  switch (c) {
  case '|':  // or
  case ',':  // comma
  case '-':  // minus
  case '=':  // equal
  case '~':  // not
  case '&':  // and
  case '*':  // star / mult
  case '/':  // div
  case '\\': // mod
  case '+':  // plus
  case '>':  // more
  case '<':  // less
  case '@':  // at
  case '%':  // per
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Generic Parsing
//===----------------------------------------------------------------------===//

/// scan forwards for the character c.
bool next(ParseCursor &cursor, char pattern, std::size_t offset = 1) {
  if (offset > cursor.remaining()) {
    return false;
  }
  return cursor.get(offset) == pattern;
}

// bool next(ParseCursor &cursor, const char *pattern, std::size_t offset = 1) {
//   return next(cursor, pattern, strlen(pattern), offset);
// }

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
  using namespace std::string_literals;
  if (!match(cursor, pattern)) {
    abort(cursor, ("expected: "s) + pattern);
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

void skipWhitespace(ParseCursor &cursor) { zero_plus(cursor, isSpace); }

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
// Symbols
//===----------------------------------------------------------------------===//

// Parse a Smalltalk Name symbol.  I.e. (Alpha AlNum+). E.g. Fry32
OptIdentifier parseNameSym(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (match(cursor, isAlpha) && zero_plus(cursor, isAlnum)) {
    return {{mkloc(start, cursor), cursor.subStringFrom(start)}};
  }

  cursor = start;
  return std::nullopt;
}

bool isBinarySelector(const std::string &str) { return isOperator(str[0]); }

bool isKeywordSelector(const std::string &str) {
  return (str.length() != 0) && (str[str.length() - 1] == ':');
}

bool isUnarySelector(const std::string &str) {
  return !isBinarySelector(str) && !isKeywordSelector(str);
}

OptIdentifier parseUnarySym(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;
  auto sym = parseNameSym(cursor);
  if (sym && isUnarySelector(sym->value) && !match(cursor, ':')) {
    return sym;
  }
  cursor = start;
  return std::nullopt;
}

// Parse a Smalltalk Operator symbol.  An operator is a string of operator
// characters. E.g. ++-
OptIdentifier parseOperatorSym(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;
  if (one_plus(cursor, isOperator)) {
    return {{mkloc(start, cursor), cursor.subStringFrom(start)}};
  }
  cursor = start;
  return std::nullopt;
}

// Parse a Smalltalk Keyword symbol. E.g. send:
OptIdentifier parseKeywordSym(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;
  if (match(cursor, isAlpha) && one_plus(cursor, isAlnum) &&
      match(cursor, ':')) {
    return {{mkloc(start, cursor), cursor.subStringFrom(start)}};
  }
  cursor = start;
  return std::nullopt;
}

/// Parse any type of symbol
OptIdentifier parseSymbol(ParseCursor &cursor) {

  OptIdentifier sym = std::nullopt;

  sym = parseKeywordSym(cursor);
  if (sym)
    return sym;

  sym = parseOperatorSym(cursor);
  if (sym)
    return sym;

  sym = parseUnarySym(cursor);
  if (sym)
    return sym;

  return std::nullopt;
}

OptIdentifier parseIdentifier(ParseCursor &cursor) {
  return parseNameSym(cursor);
}

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

ExprPtr parseLitExpr(ParseCursor &cursor);

/// Can parse a float or an integer
ExprPtr parseLitNumberExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  match(cursor, '-') || match(cursor, '+');

  if (one_plus(cursor, isDigit)) {
    auto tryfloat = cursor;
    if (match(cursor, '.') && one_plus(cursor, isDigit)) {
      auto location = mkloc(start, cursor);
      auto value = std::stof(cursor.subStringFrom(start));
      return std::make_unique<LitFloatExpr>(location, value);
    }
    // Do not eat a matched decimal point `.` if it is an integer
    cursor = tryfloat;
    auto location = mkloc(start, cursor);
    auto value = std::stoi(cursor.subStringFrom(start));
    return std::make_unique<LitIntegerExpr>(location, value);
  }

  cursor = start;
  return nullptr;
}

LitIntegerExprPtr parseLitIntegerExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  match(cursor, '-') || match(cursor, '+');

  if (one_plus(cursor, isDigit)) {
    auto location = mkloc(start, cursor);
    auto value = std::stoi(cursor.subStringFrom(start));
    return std::make_unique<LitIntegerExpr>(location, value);
  }

  cursor = start;
  return nullptr;
}

LitFloatExprPtr parseLitFloatExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  match(cursor, '-') || match(cursor, '+');

  if (!one_plus(cursor, isDigit)) {
    cursor = start;
    return nullptr;
  }

  if (!match(cursor, '.')) {
    cursor = start;
    return nullptr;
  }

  if (!one_plus(cursor, isDigit)) {
    cursor = start;
    return nullptr;
  }

  auto location = mkloc(start, cursor);
  auto value = std::stof(cursor.subStringFrom(start));
  return std::make_unique<LitFloatExpr>(location, value);
}

LitStringExprPtr parseLitStringExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (match(cursor, '\'')) {
    // do not skip() over comments inside of strings
    do {
      ++cursor;
      if (cursor.atEnd()) {
        abort(start, "un-closed literal string");
      } else if (*cursor == '\'') {
        ++cursor;
        auto location = mkloc(start, cursor);
        auto value = cursor.subStringFrom(start);
        return std::make_unique<LitStringExpr>(location, value);
      } else if (*cursor == '\\') {
        ++cursor;
        if (cursor.atEnd()) {
          abort(cursor, "un-closed literal string on escape character");
        }
      }
    } while (true);
  }

  cursor = start;
  return nullptr;
}

LitSymbolExprPtr parseLitSymbolExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  LitSymbolExprPtr litSymbol;

  if (match(cursor, '#')) {
    auto symbol = parseSymbol(cursor);
    if (!symbol) {
      abort(cursor, "Error parsing literal symbol");
    }
    auto location = mkloc(start, cursor);
    return std::make_unique<LitSymbolExpr>(location, symbol->value);
  }

  cursor = start;
  return nullptr;
}

/// <lit-array> ::= '#(' <literal>* ')'
/// note: literal arrays may only hold other literals.
LitArrayExprPtr parseLitArrayExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, "#(")) {
    cursor = start;
    return nullptr;
  }

  LitArrayExprPtr litArray = std::make_unique<LitArrayExpr>();

  while (true) {
    skip(cursor);
    auto element = parseLitExpr(cursor);
    if (!element)
      break;
    litArray->elements.push_back(std::move(element));
  }

  if (!matchNext(cursor, ')')) {
    abort(cursor, "expected ) terminating array literal");
  }

  litArray->location = mkloc(start, cursor);
  return litArray;
}

/// Parse any literal type.
ExprPtr parseLitExpr(ParseCursor &cursor) {

  ExprPtr expr = nullptr;

  expr = parseLitArrayExpr(cursor);
  if (expr)
    return expr;

  expr = parseLitStringExpr(cursor);
  if (expr)
    return expr;

  expr = parseLitNumberExpr(cursor);
  if (expr)
    return expr;

  expr = parseLitSymbolExpr(cursor);
  if (expr)
    return expr;

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Variable List
//===----------------------------------------------------------------------===//

/// Parse a variable list.  e.g. | Fry Bender Leela |
std::optional<VarList> parseVarList(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor.pos();

  if (!match(cursor, '|')) {
    return std::nullopt;
  }

  VarList list;

  while (true) {
    // ID
    auto id = parseIdentifier(cursor);
    if (id) {
      list.elements.push_back(*id);
    } else {
      abort(cursor, "Invalid character in variable list");
    }

    // |
    if (matchNext(cursor, '|')) {
      list.location.end = cursor.pos();
      break;
    }

    // EOF
    if (cursor.atEnd()) {
      abort(cursor, "Unexpected end of variable list");
    }
  }

  list.location = mkloc(start, cursor);
  return {list};
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

ExprPtr parseStatement(ParseCursor &cursor);
ExprPtr parseExpr(ParseCursor &cursor);
BlockExprPtr parseBlockExpr(ParseCursor &cursor);

IdentifierExprPtr parseIdentifierExpr(ParseCursor &cursor) {
  auto id = parseIdentifier(cursor);
  if (id) {
    return std::make_unique<IdentifierExpr>(id->location, id->value);
  }
  return nullptr;
}

/// <expr-group> ::= '(' <expr> ')'
ExprPtr parseExprGroup(ParseCursor &cursor) {
  if (!matchNext(cursor, '(')) {
    return nullptr;
  }
  auto subexpr = parseExpr(cursor);
  expectNext(cursor, ')');
  return subexpr;
}

ExprPtr parsePrimary(ParseCursor &cursor);

ExprPtr parseUnaryExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto recv = parsePrimary(cursor);
  if (!recv) {
    return nullptr;
  }

  while (true) {
    auto sym = parseUnarySym(cursor);
    if (!sym) {
      return recv;
    }
    auto expr = std::make_unique<SendExpr>();
    expr->location = mkloc(start, cursor);
    expr->selector.push_back(*sym);
    expr->parameters.push_back(std::move(recv));
    recv = std::move(expr);
  }
}

ExprPtr parseBinaryExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto recv = parseUnaryExpr(cursor);
  if (!recv) {
    return nullptr;
  }

  while (true) {
    auto sym = parseOperatorSym(cursor);
    if (!sym) {
      return recv;
    }
    auto rhs = parseUnaryExpr(cursor);

    auto expr = std::make_unique<SendExpr>();
    expr->location = mkloc(start, cursor);
    expr->selector.push_back(*sym);
    expr->parameters.push_back(std::move(recv));
    expr->parameters.push_back(std::move(rhs));
    recv = std::move(expr);
  }

  return recv;
}

ExprPtr parseKeywordExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto recv = parseBinaryExpr(cursor);
  if (!recv) {
    return nullptr;
  }

  auto sym = parseKeywordSym(cursor);
  if (!sym) {
    return recv;
  }

  // Parse the keywords
  auto expr = std::make_unique<SendExpr>();
  expr->parameters.push_back(std::move(recv));
  while (sym) {
    auto recv = parseBinaryExpr(cursor);
    expr->selector.push_back(*sym);
    expr->parameters.push_back(std::move(recv));
    sym = parseKeywordSym(cursor);
  }
  expr->location = mkloc(start, cursor);
  return expr;
}

ExprPtr parseSendExpr(ParseCursor &cursor) { return parseKeywordExpr(cursor); }

/// <primary> ::= <literal> | <identifier> | <expr-group>
ExprPtr parsePrimary(ParseCursor &cursor) {
  skip(cursor);

  if (!cursor.more()) {
    return nullptr;
  }

  ExprPtr expr = nullptr;

  expr = parseLitExpr(cursor);
  if (expr)
    return expr;

  expr = parseIdentifierExpr(cursor);
  if (expr)
    return expr;

  expr = parseExprGroup(cursor);
  if (expr)
    return expr;

  expr = parseBlockExpr(cursor);
  if (expr)
    return expr;

  return nullptr;
}

/// <expr> ::= <primary>? | <primary>?
ExprPtr parseExpr(ParseCursor &cursor) {

  skip(cursor);

  if (!cursor.more()) {
    return nullptr;
  }

  /// period terminates the expression.
  if (*cursor == '.') {
    return nullptr;
  }

  /// paren terminates.
  if (*cursor == ')') {
    return nullptr;
  }

  auto expr = parseSendExpr(cursor);
  if (expr) {
    return expr;
  }

  return nullptr;
}

/// <identifier> ':='
OptIdentifier parseAssign(ParseCursor &cursor) {
  auto start = cursor;
  auto id = parseIdentifier(cursor);
  if (!id) {
    cursor = start;
    return std::nullopt;
  }
  if (!matchNext(cursor, ":=")) {
    cursor = start;
    return std::nullopt;
  }
  return id;
}

/// <assignment> ::= ( <identifer> ':=' )+ <statement>
AssignmentExprPtr parseAssignmentStatement(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto id = parseAssign(cursor);
  if (!id) {
    cursor = start;
    return nullptr;
  }

  auto expr = std::make_unique<AssignmentExpr>();

  expr->identifiers.push_back(*id);

  while (true) {
    auto id = parseAssign(cursor);
    if (!id)
      break;
    expr->identifiers.push_back(*id);
  }

  auto value = parseExpr(cursor);

  if (!value) {
    abort(cursor, "expected expression following := operator");
  }

  expr->value = std::move(value);
  return expr;
}

ReturnExprPtr parseReturnStatement(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '^')) {
    return nullptr;
  }

  auto expression = parseExpr(cursor);
  matchNext(cursor, '.');
  auto location = mkloc(start, cursor);
  return std::make_unique<ReturnExpr>(location, std::move(expression));
}

/// <statement> ::= ( <assignment> | <expr>  | <return> )
ExprPtr parseStatement(ParseCursor &cursor) {
  ExprPtr expr = nullptr;

  expr = parseAssignmentStatement(cursor);
  if (expr) {
    return expr;
  }

  expr = parseReturnStatement(cursor);
  if (expr) {
    return expr;
  }

  expr = parseExpr(cursor);
  if (expr) {
    return expr;
  }

  return nullptr;
}

/// <statement-list> ::= <statement>*
ExprPtrList parseStatementList(ParseCursor &cursor) {
  ExprPtrList list;
  while (true) {
    auto expr = parseStatement(cursor);
    if (!expr)
      break;
    list.push_back(std::move(expr));
    match(cursor, ".");
  }
  return list;
}

//===----------------------------------------------------------------------===//
// Blocks Expressions
//===----------------------------------------------------------------------===//

/// block arguments
OptVarList parseOptionalBlockArguments(ParseCursor &cursor) {
  skip(cursor);

  auto c = cursor;
  if (!match(c, ':')) {
    return std::nullopt;
  }

  VarList varList;

  OptIdentifier id;
  while (id = parseIdentifier(cursor)) {
    varList.elements.push_back(*id);
  }

  expectNext(c, '|');

  cursor = c;
  return varList;
}

BlockExprPtr parseBlockExpr(ParseCursor &cursor) {

  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '[')) {
    return nullptr;
  }

  auto block = std::make_unique<BlockExpr>();

  // Valid forms, arguments followed by locals
  // [ a + b ]
  // [ :a | a + b ]
  // [ | b | a + b ]
  // [ :a | | b | a + b ]

  skip(cursor);

  // if (cursor.more() && *cursor == ':') {
  //   do {
  //     ++cursor;
  //     auto sym = parseIdentifier(cursor);
  //     if (!sym) {
  //       abort(cursor, "Error parsing block argument, colon but no symbol");
  //     }
  //     block->parameters.push_back(*sym);
  //     skip(cursor);
  //   } while (cursor.more() && *cursor == ':');

  //   skip(cursor);

  //   if (*cursor != '|') {
  //     abort(cursor, "Unterminated block argument sequence");
  //   }
  //   ++cursor;
  // }

  if (match(cursor, ':')) {
    while (true) {
      auto sym = parseIdentifier(cursor);
      if (!sym) {
        abort(cursor, "Error parsing block argument, colon but no symbol");
      }
      block->parameters.push_back(*sym);
      if (matchNext(cursor, '|')) {
        break;
      }
      expectNext(cursor, ':');
    }
  }

  // Block locals
  block->locals = parseVarList(cursor);

  // Block body
  block->body = parseStatementList(cursor);

  expectNext(cursor, ']');

  block->location = mkloc(start, cursor);

  return block;
}

//===----------------------------------------------------------------------===//
// Method
//===----------------------------------------------------------------------===//

std::optional<std::unique_ptr<Method>> parseMethod(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto method = std::make_unique<Method>();

  //
  // Signature
  //

  OptIdentifier id = std::nullopt;
  if (id = parseKeywordSym(cursor)) {
    // keyword signature
    while (id) {
      OptIdentifier param = parseNameSym(cursor);
      if (!param) {
        abort(cursor, "Keyword method signature missing parameter");
      }
      method->selector.push_back(*id);
      method->parameters.push_back(*param);

      id = parseKeywordSym(cursor);
    }
  } else if (id = parseOperatorSym(cursor)) {
    // binary signature
    OptIdentifier param = parseNameSym(cursor);
    if (!param) {
      abort(cursor, "Binary method signature missing parameter");
    }
    method->selector.push_back(*id);
    method->parameters.push_back(*param);

  } else if (id = parseNameSym(cursor)) {
    // unary signature
    method->selector.push_back(*id);
  } else {
    cursor = start;
    return std::nullopt;
  }

  // = (
  // = primitive
  expectNext(cursor, '=');
  if (matchNext(cursor, "primitive")) {
    method->location = mkloc(start, cursor);
    return {std::move(method)};
  }
  expectNext(cursor, '(');

  // | a b c |
  method->locals = parseVarList(cursor);

  // Body parsing
  method->body = parseStatementList(cursor);

  // )
  expectNext(cursor, ')');

  method->location = mkloc(start, cursor);

  return {std::move(method)};
}

//===----------------------------------------------------------------------===//
// Class
//===----------------------------------------------------------------------===//

/// three or more dashes, separating class-fields and class-methods from
/// instance-fields and instance-methods.
///
bool separator(ParseCursor &cursor) { return n_plus(cursor, 3, '-'); }

std::unique_ptr<Klass> parseKlass(ParseCursor &cursor) {

  auto klass = std::make_unique<Klass>();
  auto start = cursor.pos();

  // ClassName
  auto id = parseIdentifier(cursor);
  if (!id) {
    abort(cursor, "Invalid class name");
  }
  klass->name = *id;

  // =
  expectNext(cursor, '=');

  // superklass?
  auto super = parseIdentifier(cursor);
  if (super) {
    klass->super = *super;
  }

  // (
  expectNext(cursor, '(');

  // Parse members

  while (true) {

    skip(cursor);

    // | A B C |
    if (auto fields = parseVarList(cursor); fields) {
      if (klass->fields) {
        abort(cursor,
              "Class " + klass->name.value + " has two field declarations");
      }
      klass->fields = *fields;
    }

    // Methods
    if (auto method = parseMethod(cursor); method) {
      klass->methods.push_back(std::move(*method));
    }

    // )
    if (matchNext(cursor, ')')) {
      break;
    }

    // EOF
    if (cursor.atEnd()) {
      abort(cursor, "Unexpected end of class" + klass->name.value);
    }
  }

  klass->location = mkloc(start, cursor);
  return klass;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

std::unique_ptr<Module> parseModule(ParseCursor &cursor) {
  auto module = std::make_unique<Module>(cursor.loc());
  auto start = cursor.pos();

  skip(cursor);
  while (cursor.more()) {
    module->klasses.push_back(parseKlass(cursor));
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
