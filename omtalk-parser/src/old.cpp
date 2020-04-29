#include "ParseCursor.h"
#include <omtalk/parser/AST.h>
#include <omtalk/parser/Lex.h>
#include <omtalk/parser/Location.h>
#include <omtalk/parser/Parser.h>

namespace {

constexpr bool is_whitespace(const char c) {
  switch (c) {
  case ' ':
  case '\n':
  case '\t':
    return true;
  default:
    return false;
  }
}

constexpr bool is_alpha(const char x) {
  return 'a' <= x && x <= 'z' || 'A' <= x && x <= 'Z';
}

constexpr bool is_digit(const char x) { return '0' <= x && x <= '9'; }

constexpr bool is_alnum(const char x) { return is_alpha(x) || is_digit(x); }

constexpr bool is_operator(char c) {
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

inline void skip_whitespace(ParseCursor &cursor) {
  while (cursor.offset() < cursor.in_length() && is_whitespace(*cursor)) {
    ++cursor;
  }
}

inline ast::Comment parse_comment(ParseCursor &cursor) {
  assert(cursor.get() == '"');
  const ParseCursor start = cursor;
  do {
    ++cursor;
    if (cursor.end()) {
      throw std::exception();
    } else if (*cursor == '\"') {
      ++cursor;
      break;
    } else if (*cursor == '\\') {
      ++cursor;
      if (cursor.end()) {
        throw std::exception();
      }
    }
  } while (true);
  return ast::Comment(start.location(), cursor.location(),
                      std::string(start.in_iter(), cursor.in_iter()));
}

inline void skip(ParseCursor &cursor) {
  bool cont = true;
  while (true) {
    if (cursor.end()) {
      break;
    } else if (*cursor == '\"') {
      parse_comment(cursor);
    } else if (is_whitespace(*cursor)) {
      skip_whitespace(cursor);
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

void expect_next(ParseCursor &cursor, char c) {
  skip(cursor);
  expect(cursor, c);
}

void expect(ParseCursor &cursor, const std::string &s) {
  for (auto x : s) {
    if (*cursor == x) {
      ++cursor;
    } else {
      throw false;
    }
  }
}

void expect_next(ParseCursor &cursor, std::string s) {
  skip(cursor);
  expect(cursor, s);
}

inline void discard_until(ParseCursor &cursor, char c) {
  while (*cursor != c) {
    ++cursor;
  }
}

inline ast::Symbol parse_symbol(ParseCursor &cursor) {
  const ParseCursor start = cursor;
  if (cursor.end()) {
    throw std::exception();
  }
  if (ast::is_operator(*cursor)) {
    ++cursor;
  } else {
    while (!cursor.end()) {
      if (*cursor == ':') {
        ++cursor;
        break;
      }
      if (!is_alnum(*cursor)) {
        break;
      }
      ++cursor;
    }
  }
  return ast::Symbol(start.location(), cursor.location(),
                     std::string(start.in_iter(), cursor.in_iter()));
}

inline ast::String parse_string(ParseCursor &cursor) {
  assert(cursor.get() == '\'');
  ++cursor;
  const ParseCursor start = cursor;
  do {
    ++cursor;
    if (cursor.end()) {
      throw std::exception();
    } else if (*cursor == '\'') {
      break;
    } else if (*cursor == '\\') {
      ++cursor;
      if (cursor.end()) {
        throw std::exception();
      }
    }
  } while (true);
  return ast::String(start.location(), cursor.location(),
                     std::string(start.in_iter(), cursor.in_iter()));
}

inline ast::Integer parse_integer(ParseCursor &cursor) {
  assert(is_digit(cursor.get()));
  const ParseCursor start = cursor;
  do {
    ++cursor;
    if (cursor.end()) {
      break;
    } else if (is_digit(cursor.get())) {
      ++cursor;
    }
  } while (true);
  return ast::Integer(
      start.location(), cursor.location(),
      std::stoi(std::string(start.in_iter(), cursor.in_iter())));
}

inline ast::Symbol parse_super(ParseCursor &cursor) {
  return parse_symbol(cursor);
}

inline ast::SymbolList parse_locals(ParseCursor &cursor) {
  expect(cursor, '|');
  ast::SymbolList locals;
  for (bool cont = true; cont;) {
    skip(cursor);
    if (cursor.end()) {
      throw std::exception();
    } else if (*cursor == '|') {
      ++cursor;
      break;
    } else if (is_alpha(*cursor)) {
      locals.push_back(std::make_shared<ast::Symbol>(parse_symbol(cursor)));
    } else {
      throw std::exception();
    }
  }
  return locals;
}

inline ast::Symbol parse_identifier(ParseCursor &cursor) {
  ast::Symbol symbol = parse_symbol(cursor);
  assert(ast::symbol_kind(symbol) == ast::SymbolKind::IDENTIFIER);
  return symbol;
}

ast::BinarySignature parse_binary_signature(ParseCursor &cursor,
                                            ast::Symbol symbol) {
  ast::BinarySignature sig;
  sig.symbol(symbol);
  skip(cursor);
  sig.argument(parse_identifier(cursor));
  return sig;
}

ast::KeywordSignature parse_keyword_signature(ParseCursor &cursor,
                                              ast::Symbol &&symbol) {
  ast::KeywordSignature signature;
  ast::Symbol keyword = symbol;
  ast::Symbol argument = parse_symbol(cursor);

  assert(ast::symbol_kind(keyword) == ast::SymbolKind::KEYWORD);
  assert(ast::symbol_kind(argument) == ast::SymbolKind::IDENTIFIER);
  signature.arguments().push_back({std::move(keyword), std::move(argument)});

  skip(cursor);

  while (!cursor.end()) {
    if (*cursor == '=') {
      break;
    } else {
      ast::Symbol keyword = parse_symbol(cursor);
      skip(cursor);
      ast::Symbol argument = parse_symbol(cursor);
      skip(cursor);

      assert(ast::symbol_kind(keyword) == ast::SymbolKind::KEYWORD);
      assert(ast::symbol_kind(argument) == ast::SymbolKind::IDENTIFIER);
      signature.arguments().push_back(
          {std::move(keyword), std::move(argument)});
    }
  }

  return signature;
}

inline ast::SignaturePtr parse_signature(ParseCursor &cursor) {
  ast::Symbol symbol = parse_symbol(cursor);
  switch (ast::symbol_kind(symbol)) {
  case ast::SymbolKind::IDENTIFIER:
    return std::make_shared<ast::UnarySignature>(std::move(symbol));
  case ast::SymbolKind::OPERATOR:
    return std::make_shared<ast::BinarySignature>(
        parse_binary_signature(cursor, symbol));
  case ast::SymbolKind::KEYWORD:
    skip(cursor);
    return std::make_shared<ast::KeywordSignature>(
        std::move(parse_keyword_signature(cursor, std::move(symbol))));
  default:
    return nullptr;
  }
}

ast::ExpressionPtr parse_expression(ParseCursor &cursor);

inline ast::SymbolExpression parse_symbol_expression(ParseCursor &cursor,
                                                     ast::Symbol &&symbol) {
  ast::SymbolExpression expression;
  assert(ast::symbol_kind(symbol) == ast::SymbolKind::IDENTIFIER);
  expression.symbol(symbol);
  return expression;
}

inline ast::UnaryExpression
parse_unary_expression(ParseCursor &cursor, ast::ExpressionPtr &&receiver,
                       ast::Symbol &&message) {
  assert(ast::symbol_kind(message) == ast::SymbolKind::IDENTIFIER);
  ast::UnaryExpression expression;
  expression.receiver(receiver);
  expression.message(message);
  return expression;
}

inline ast::BinaryExpression
parse_binary_expression(ParseCursor &cursor, ast::ExpressionPtr &&receiver,
                        ast::Symbol &&symbol) {
  assert(ast::symbol_kind(symbol) == ast::SymbolKind::OPERATOR);

  ast::BinaryExpression expression;
  expression.receiver(receiver);
  expression.message(symbol);
  expression.argument(parse_expression(cursor));

  return expression;
}

inline ast::UnitExpression parse_unit_expression(ParseCursor &cursor) {
  ast::UnitExpression expression;

  return expression;
}

inline ast::ExpressionPtr parse_expression(ParseCursor &cursor) {
  // expr := (unary | binary | unit)

  // unit := '(' expr ')'
  // symbol := SYMBOL
  // unary := (unary | binary | unit | symbol) symbol
  // binary := (unary | binary | unit | symbol) OP (unary | binary | unit |
  // symbol)

  // keyword := expr KEYWORD: expr

  ast::ExpressionPtr receiver;

  if (*cursor == ')') {
    assert(false);
  } else if (*cursor == '.') {
    assert(false);
  } else if (*cursor == '(') {
    // Unit expression
    receiver =
        std::make_shared<ast::UnitExpression>(parse_unit_expression(cursor));
  } else {
    ast::Symbol symbol = parse_symbol(cursor);
    assert(ast::symbol_kind(symbol) == ast::SymbolKind::IDENTIFIER);
    receiver = std::make_shared<ast::SymbolExpression>(
        parse_symbol_expression(cursor, std::move(symbol)));
  }

  skip(cursor);
  if (*cursor == '.' || *cursor == ')') {
    return receiver;
  }

  // ast::ExpressionPtr expr = nullptr;
  ast::Symbol symbol = parse_symbol(cursor);
  switch (ast::symbol_kind(symbol)) {
  case ast::SymbolKind::IDENTIFIER: {
    auto e = std::make_shared<ast::UnaryExpression>();
    e->receiver(std::move(receiver));
    e->message(std::move(symbol));
    receiver = e;
  }
  case ast::SymbolKind::KEYWORD:
    // return std::make_shared<ast::KeywordSignature>(
    //     std::move(parse_keyword_signature(cursor, std::move(message))));
  case ast::SymbolKind::OPERATOR:
    receiver = std::make_shared<ast::Expression>(
        parse_binary_signature(cursor, receiver, symbol));
  default:
    return nullptr;
  }

  return ast::Expression{parse_symbol(cursor)};
  return std::make_shared<ast::UnaryExpression>(parse_unary_expression(cursor));
}

inline ast::Body parse_body(ParseCursor &cursor) {
  ast::Body body;
  if (*cursor == '(') {
  } else if (*cursor == 'p') {
  }
  assert(*cursor == '(');
  discard_until(cursor, ')'); // TODO: This throws away the body.
  return ast::Body();
}

inline ast::Method parse_method(ParseCursor &cursor) {
  ast::Method method;
  method.signature(parse_signature(cursor));
  expect_next(cursor, '=');
  skip(cursor);
  assert(*cursor == '(');
  method.body(parse_body(cursor));
  return method;
}

inline ast::Clazz parse_clazz(ParseCursor &cursor) {
  ast::Clazz clazz;
  const ParseCursor start = cursor;
  clazz.name(parse_symbol(cursor));
  skip(cursor);
  expect_next(cursor, '=');
  skip(cursor);
  if (is_alpha(*cursor)) {
    clazz.super(parse_super(cursor));
    skip(cursor);
  }
  expect_next(cursor, '(');
  for (bool cont = true; cont;) {
    skip(cursor);
    if (cursor.end()) {
      throw std::exception();
    } else if (*cursor == '|') {
      clazz.locals() = parse_locals(cursor);
    } else if (*cursor == ')') {
      cont = false;
    } else {
      clazz.methods().push_back(
          std::make_shared<ast::Method>(parse_method(cursor)));
    }
  }
  expect_next(cursor, ')');
  clazz.location({start.location(), cursor.location()});
  return clazz;
}

inline ast::Root parse(ParseCursor &cursor) {
  bool cont = true;
  while (cont) {
    skip(cursor);
    if (cursor.end()) {
      break;
    }

    if (*cursor == '\"') {
      assert(0);
    } else if (is_alpha(*cursor)) {
      parse_symbol(cursor);
    }
  }
  return ast::Root();
}

} // namespace

namespace omtalk {
namespace parser {}
} // namespace omtalk
