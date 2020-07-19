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

/// Returns if the symbol is a smalltalk operator.
constexpr bool isOperator(char c) {
  switch (c) {
  case '-':  // minus
  case ',':  // comma
  case '@':  // at
  case '*':  // star
  case '/':  // div
  case '\\': // mod
  case '&':  // and
  case '%':  // per
  case '+':  // plus
  case '<':  // less
  case '=':  // equal
  case '>':  // more
  case '|':  // or
  case '~':  // not
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Generic Parsing
//===----------------------------------------------------------------------===//

/// Scan forwards as long as the predicate matches.
bool match(ParseCursor &cursor, bool (*pred)(char)) {
  if (cursor.more() && pred(*cursor)) {
    ++cursor;
    return true;
  }
  return false;
}

/// Scan forwards for the character c.
bool match(ParseCursor &cursor, char pattern) {
  if (cursor.more() && *cursor == pattern) {
    ++cursor;
    return true;
  }
  return false;
}

/// Scan forwards for a string `pattern`.
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

/// Scan forwards for a string `pattern`.
bool match(ParseCursor &cursor, const char *pattern) {
  return match(cursor, pattern, strlen(pattern));
}

/// Match any character that is not the pattern.
template <typename P>
bool match_not(ParseCursor &cursor, P pattern) {
  auto copy = cursor;
  return !match(copy, pattern);
}

/// Matches the pattern.  If it does not match, aborts.
template <typename P>
void expect(ParseCursor &cursor, const P &pattern) {
  using namespace std::string_literals;
  if (!match(cursor, pattern)) {
    abort(cursor, ("expected: "s) + pattern);
  }
}

/// Match pattern N times exactly.
template <typename P>
bool n_times(ParseCursor &cursor, int count, const P &pattern) {
  auto start = cursor;
  for (unsigned i = 0; i < count; ++i) {
    if (!match(cursor, pattern)) {
      cursor = start;
      return false;
    }
  }
  return true;
}

/// Match pattern n or more times.
template <typename P>
bool n_plus(ParseCursor &cursor, int count, const P &pattern) {
  if (!n_times(cursor, count, pattern)) {
    return false;
  }
  do {
  } while (match(cursor, pattern));
  return true;
}

/// Match pattern zero or more times.
template <typename P>
bool zero_plus(ParseCursor &cursor, const P &pattern) {
  return n_plus(cursor, 0, pattern);
}

/// Match  pattern one or more times.
template <typename P>
bool one_plus(ParseCursor &cursor, const P &pattern) {
  return n_plus(cursor, 1, pattern);
}

//===----------------------------------------------------------------------===//
// Comments and Whitespace
//===----------------------------------------------------------------------===//

void skipWhitespace(ParseCursor &cursor) { zero_plus(cursor, isSpace); }

void parseComment(ParseCursor &cursor) {
  skipWhitespace(cursor);
  if (!match(cursor, '"')) {
    return;
  }

  do {
    ++cursor;
    if (cursor.atEnd()) {
      abort(cursor, "un-closed comment");
    } else if (*cursor == '\"') {
      ++cursor;
      break;
    } else if (*cursor == '\\') {
      ++cursor;
      if (cursor.atEnd()) {
        abort(cursor, "un-closed comment on escape character");
      }
    }
  } while (true);
}

void parseDirective(ParseCursor &cursor) {
  skipWhitespace(cursor);
  if (!match(cursor, '@')) {
    return;
  }

  do {
    ++cursor;
  } while (cursor.more() && cursor.get() != '\n');
}

void skip(ParseCursor &cursor) {
  while (true) {
    skipWhitespace(cursor);
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

/// Skip white space and then expect the pattern.  Aborts process if the pattern
/// does not match.
template <typename P>
void expectNext(ParseCursor &cursor, const P &pattern) {
  skip(cursor);
  expect(cursor, pattern);
}

/// Skip white space and then match the pattern.
template <typename P>
bool matchNext(ParseCursor &cursor, const P &pattern) {
  skip(cursor);
  return match(cursor, pattern);
}

//===----------------------------------------------------------------------===//
// Symbols
//===----------------------------------------------------------------------===//

/// Checks if a string is a binary selector, e.g. `+`.
bool isBinarySelector(const std::string &str) {
  if (str.size()) {
    // Binary selectors start with operators
    return isOperator(str[0]);
  }
  return false;
}

/// Checks if a string is a keyword selector, e.g. `mykeyword:`.
bool isKeywordSelector(const std::string &str) {
  return (str.length() != 0) && (str[str.length() - 1] == ':');
}

/// Checks if a string is a unary selector.
bool isUnarySelector(const std::string &str) {
  return !isBinarySelector(str) && !isKeywordSelector(str);
}

/// Parse a Smalltalk Name symbol.  I.e. (Alpha AlNum+). E.g. Fry32
OptIdentifier parseNameSym(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (match(cursor, isAlpha) && zero_plus(cursor, [](const char c) {
        return isAlnum(c) || (c == '_');
      })) {
    return {{mkloc(start, cursor), cursor.subStringFrom(start)}};
  }

  cursor = start;
  return std::nullopt;
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

/// Parse any type of symbol
OptIdentifier parseAnySymbol(ParseCursor &cursor) {
  auto start = cursor;
  if (match(cursor, [](const char c) { return !isSpace(c); })) {
    return {{mkloc(start, cursor), cursor.subStringFrom(start)}};
  }
  cursor = start;
  return std::nullopt;
}

OptIdentifier parseIdentifier(ParseCursor &cursor) {
  return parseNameSym(cursor);
}

//===----------------------------------------------------------------------===//
// Special words
//===----------------------------------------------------------------------===//

ExprPtr parseSpecialExpr(ParseCursor &cursor) {
  skip(cursor);
  auto save = cursor;
  auto id = parseIdentifier(cursor);

  if (id->value == "nil") {
    return makeNilExpr(id->location);
  }

  if (id->value == "true") {
    return makeBoolExpr(id->location, true);
  }

  if (id->value == "false") {
    return makeBoolExpr(id->location, false);
  }

  if (id->value == "self") {
    return makeSelfExpr(id->location);
  }

  if (id->value == "super") {
    return makeSuperExpr(id->location);
  }

  if (id->value == "system") {
    return makeSystemExpr(id->location);
  }

  cursor = save;
  return nullptr;
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
      return makeFloatExpr(location, value);
    }
    // Do not eat a matched decimal point `.` if it is an integer
    cursor = tryfloat;
    auto location = mkloc(start, cursor);
    auto value = std::stoi(cursor.subStringFrom(start));
    return makeIntegerExpr(location, value);
  }

  cursor = start;
  return nullptr;
}

StringExprPtr parseStringExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '\'')) {
    return nullptr;
  }

  std::string value;

  while (true) {
    if (!cursor.more()) {
      abort(start, "unclosed literal string");
    }

    if (*cursor == '\'') {
      ++cursor;
      break;
    }

    if (*cursor == '\\') {
      ++cursor;
      if (!cursor.more()) {
        abort(cursor, "incomplete string escape sequence");
      }
      switch (*cursor) {
      case 'n':
        value += '\n';
        break;
      case 't':
        value += '\t';
        break;
      case '\'':
        value += '\'';
        break;
      default:
        abort(cursor, "unrecognized string escape sequence");
        break;
      }
    } else {
      value += *cursor;
    }

    ++cursor;
  }

  return makeStringExpr(mkloc(start, cursor), std::move(value));
}

SymbolExprPtr parseSymbolExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (match(cursor, '#')) {
    SymbolExprPtr litSymbol;
    auto symbol = parseSymbol(cursor);
    if (!symbol) {
      abort(cursor, "Error parsing literal symbol");
    }
    auto location = mkloc(start, cursor);
    return makeSymbolExpr(location, symbol->value);
  }

  cursor = start;
  return nullptr;
}

/// <lit-array> ::= '#(' <literal>* ')'
/// note: literal arrays may only hold other literals.
ArrayExprPtr parseArrayExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, "#(")) {
    cursor = start;
    return nullptr;
  }

  auto litArray = makeArrayExpr();

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

  expr = parseArrayExpr(cursor);
  if (expr)
    return expr;

  expr = parseStringExpr(cursor);
  if (expr)
    return expr;

  expr = parseLitNumberExpr(cursor);
  if (expr)
    return expr;

  expr = parseSymbolExpr(cursor);
  if (expr)
    return expr;

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Variable List
//===----------------------------------------------------------------------===//

bool matchVarListBar(ParseCursor &cursor) {
  skip(cursor);
  auto save = cursor;
  auto sym = parseOperatorSym(cursor);
  if (!sym || sym->value != "|") {
    cursor = save;
    return false;
  }
  return true;
}

/// Parse a variable list.  e.g. | Fry Bender Leela |
std::optional<VarList> parseVarList(ParseCursor &cursor) {
  auto start = cursor;

  if (!matchVarListBar(cursor)) {
    return std::nullopt;
  }

  VarList list;

  while (true) {
    auto id = parseIdentifier(cursor);
    if (!id)
      break;
    list.elements.push_back(*id);
  }

  if (!matchVarListBar(cursor)) {
    abort(cursor, "Unexpected character in variable list.");
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

AssignmentExprPtr parseAssignmentExpr(ParseCursor &cursor);

IdentifierExprPtr parseIdentifierExpr(ParseCursor &cursor) {
  auto id = parseIdentifier(cursor);
  if (!id) {
    return nullptr;
  }
  return makeIdentifierExpr(id->location, id->value);
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
    auto expr = makeSendExpr();
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

    auto expr = makeSendExpr();
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
  auto expr = makeSendExpr();
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

  expr = parseSpecialExpr(cursor);
  if (expr)
    return expr;

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

  ExprPtr expr;

  expr = parseAssignmentExpr(cursor);
  if (expr) {
    return expr;
  }

  expr = parseSendExpr(cursor);
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
AssignmentExprPtr parseAssignmentExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto id = parseAssign(cursor);
  if (!id) {
    cursor = start;
    return nullptr;
  }

  auto expr = makeAssignmentExpr();

  expr->identifier = std::move(*id);

  auto value = parseExpr(cursor);
  if (!value) {
    abort(cursor, "expected expression following := operator");
  }

  expr->value = std::move(value);
  expr->location = mkloc(start, cursor);
  return expr;
}

ReturnExprPtr parseReturnExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '^')) {
    return nullptr;
  }

  auto expression = parseExpr(cursor);
  auto location = mkloc(start, cursor);
  return makeReturnExpr(location, std::move(expression));
}

NonlocalReturnExprPtr parseNonlocalReturnExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '^')) {
    return nullptr;
  }

  auto expression = parseExpr(cursor);
  auto location = mkloc(start, cursor);
  return makeNonlocalReturnExpr(location, std::move(expression));
}

//===----------------------------------------------------------------------===//
// Blocks Expressions
//===----------------------------------------------------------------------===//

/// Parse the parameters to a block
/// Valid forms, parameters followed by locals
IdentifierList parseBlockParameters(ParseCursor &cursor) {
  skip(cursor);
  IdentifierList parameters;
  if (match(cursor, ':')) {
    while (true) {
      auto sym = parseIdentifier(cursor);
      if (!sym) {
        abort(cursor, "Error parsing block argument, colon but no symbol");
      }
      parameters.push_back(*sym);
      if (matchNext(cursor, '|')) {
        break;
      }
      expectNext(cursor, ':');
    }
  }
  return parameters;
}

ExprPtr parseBlockStatement(ParseCursor &cursor) {
  ExprPtr expr = nullptr;

  expr = parseNonlocalReturnExpr(cursor);
  if (expr) {
    return expr;
  }

  expr = parseExpr(cursor);
  if (expr) {
    return expr;
  }

  return nullptr;
}

/// Parse the body of a block. A block differs from a method body in three
/// ways:
/// 1. Return statements inside of blocks are non-local returns.  This is
/// similar to exception handling
/// 2. Blocks implicity return the value of the last statement, while methods
/// implicitly return self.
/// 3. An empty block implicitly returns nil
ExprPtrList parseBlockBody(ParseCursor &cursor) {
  ExprPtrList list;
  while (true) {
    auto expr = parseBlockStatement(cursor);
    if (!expr)
      break;
    list.push_back(std::move(expr));
    match(cursor, ".");
  }
  // emit an implicit return if needed

  ExprPtr expr = nullptr;

  if (list.size() == 0) {
    list.push_back(makeReturnExpr(cursor.loc(), makeNilExpr(cursor.loc())));
  } else if (list.back()->kind != ExprKind::NonlocalReturn) {
    auto value = std::move(list.back());
    list.pop_back();
    list.push_back(makeReturnExpr(cursor.loc(), std::move(value)));
  }
  return list;
}

/// Parse a literal block expression
/// Valid block formats:
/// [ a + b ]a
/// [ :a | a + b ]
/// [ | b | a + b ]
/// [ :a | | b | a + b ]
BlockExprPtr parseBlockExpr(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  if (!match(cursor, '[')) {
    return nullptr;
  }

  auto block = makeBlockExpr();
  block->parameters = parseBlockParameters(cursor);
  block->locals = parseVarList(cursor);
  block->body = parseBlockBody(cursor);

  expectNext(cursor, ']');

  block->location = mkloc(start, cursor);
  return block;
}

//===----------------------------------------------------------------------===//
// Method
//===----------------------------------------------------------------------===//

ExprPtr parseMethodStatement(ParseCursor &cursor) {
  ExprPtr expr = nullptr;

  expr = parseAssignmentExpr(cursor);
  if (expr) {
    return expr;
  }

  expr = parseReturnExpr(cursor);
  if (expr) {
    return expr;
  }

  expr = parseExpr(cursor);
  if (expr) {
    return expr;
  }

  return nullptr;
}

ExprPtrList parseMethodBody(ParseCursor &cursor) {
  ExprPtrList list;

  while (true) {
    auto expr = parseMethodStatement(cursor);
    if (!expr)
      break;
    list.push_back(std::move(expr));
    match(cursor, ".");
  }

  // emit an implicit return if needed

  if ((list.size() == 0) || (list.back()->kind != ExprKind::Return)) {
    list.push_back(makeReturnExpr(cursor.loc(), makeSelfExpr(cursor.loc())));
  }

  return list;
}

std::optional<MethodPtr> parseMethod(ParseCursor &cursor) {
  skip(cursor);
  auto start = cursor;

  auto method = makeMethod();

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
  method->body = parseMethodBody(cursor);

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
bool matchSeparator(ParseCursor &cursor) {
  skip(cursor);
  return n_plus(cursor, 4, '-');
}

KlassPtr parseKlass(ParseCursor &cursor) {
  auto klass = makeKlass();
  auto start = cursor.pos();

  // ClassName
  auto id = parseIdentifier(cursor);
  if (!id) {
    abort(cursor, "Invalid class name");
  }
  klass->name = *id;

  expectNext(cursor, '=');

  // superklass?
  auto super = parseIdentifier(cursor);
  if (super) {
    if (super->value == "nil") {
      klass->super = std::nullopt;
    } else {
      klass->super = *super;
    }
  } else {
    klass->super = Identifier(cursor.loc(), std::string("Object"));
  }

  expectNext(cursor, '(');

  // Fields
  auto fields = parseVarList(cursor);
  klass->fields = std::move(fields);

  // Methods
  bool parseStatic = false;
  while (true) {
    if (matchSeparator(cursor)) {
      parseStatic = true;
      break;
    }
    auto method = parseMethod(cursor);
    if (!method) {
      break;
    }
    klass->methods.push_back(std::move(*method));
  }

  // Klass Static
  if (parseStatic) {

    // Static Fields
    auto fields = parseVarList(cursor);
    klass->klassFields = std::move(fields);

    // Static Methods
    while (true) {
      auto method = parseMethod(cursor);
      if (!method) {
        break;
      }
      klass->klassMethods.push_back(std::move(*method));
    }
  }

  expectNext(cursor, ')');

  klass->location = mkloc(start, cursor);
  return klass;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

ModulePtr parseModule(ParseCursor &cursor) {
  auto module = makeModule(cursor.loc());
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

namespace {

std::string slurp(const std::string &filename) {
  std::ifstream in(filename, std::ios::in);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

} // namespace

ModulePtr omtalk::parser::parseFile(const std::string &filename) {
  auto contents = slurp(filename);
  ParseCursor cursor(filename, contents);
  auto module = parseModule(cursor);
  return module;
}
