#include <gtest/gtest.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <omtalk/ast.hpp>
#include <omtalk/parser.hpp>

using namespace omtalk;

TEST(ParseCursor, EmptyString) {
  std::string in = "";
  ParseCursor cursor(in);
  EXPECT_EQ(cursor.offset(), 0);
  EXPECT_EQ(cursor.location().line, 1);
  EXPECT_EQ(cursor.location().column, 1);
  EXPECT_TRUE(cursor.end());
}

TEST(ParseCursor, Filename) {
  std::string in = "";
  ParseCursor cursor("myfile", "mycontent");
  EXPECT_STREQ(cursor.location().filename.c_str(), "myfile");
}

TEST(ParseCursor, A) {
  std::string in = "a";
  ParseCursor cursor(in);
  EXPECT_EQ(cursor.offset(), 0);
  EXPECT_EQ(cursor.location().line, 1);
  EXPECT_EQ(cursor.location().column, 1);
  EXPECT_EQ(*cursor, 'a');
}

TEST(ParseCursor, AnB) {
  std::string in = "a\nb";
  ParseCursor cursor(in);
  ++cursor;
  ++cursor;
  EXPECT_EQ(cursor.offset(), 2);
  EXPECT_EQ(cursor.location().line, 2);
  EXPECT_EQ(cursor.location().column, 0);
  EXPECT_EQ(*cursor, 'b');
}

TEST(ParseCursor, SkipToA) {
  std::string in = "       A    B";
  ParseCursor c(in);
  skip_whitespace(c);
  EXPECT_EQ(*c, 'A');
}

TEST(Parser, SlurpResource) {
  EXPECT_STREQ("testing the tester",
               slurp(test::env->resource("test.txt")).c_str());
}

TEST(Parser, ParseSingleSymbol) {
  std::string in = "test";
  ParseCursor cursor(in);
  ast::Symbol sym = parse_symbol(cursor);
  EXPECT_STREQ(sym.c_str(), in.c_str());
}

TEST(Parser, ParseEmptyComment) {
  std::string in = "\"\"";
  ParseCursor cursor(in);
  ast::Comment comment = parse_comment(cursor);
  EXPECT_STREQ(comment.c_str(), in.c_str());
}

TEST(Parser, ParseSimpleComment) {
  std::string in = "\"test\"";
  ParseCursor cursor(in);
  ast::Comment comment = parse_comment(cursor);
  EXPECT_STREQ(comment.c_str(), in.c_str());
}

TEST(Parser, ParseEscapedDoubleQuoteComment) {
  std::string in = "\"\\\"\"";  // "\""
  ParseCursor cursor(in);
  ast::Comment comment = parse_comment(cursor);
  EXPECT_STREQ(comment.c_str(), in.c_str());
}

TEST(Parser, ParseMalformedComment) {
  std::string in = "\"";
  ParseCursor cursor(in);
  ASSERT_ANY_THROW(ast::Comment comment = parse_comment(cursor););
}

TEST(Parser, ParseSimpleString) {
  std::string in = "\'test\'";
  ParseCursor cursor(in);
  ast::String string = parse_string(cursor);
  EXPECT_STREQ(string.c_str(), "test");
}

TEST(Parser, ParseSimpleStringWithWhitespace) {
  std::string in = "\'test \'  ";
  ParseCursor cursor(in);
  ast::String string = parse_string(cursor);
  EXPECT_STREQ(string.c_str(), "test ");
}

TEST(Parser, ParseSimpleInteger) {
  std::string in = "12345";
  ParseCursor cursor(in);
  ast::Integer integer = parse_integer(cursor);
  EXPECT_EQ(integer.value(), 12345);
}

TEST(Parser, ParseSimpleIntegerWithWhitespace) {
  std::string in = "12345  ";
  ParseCursor cursor(in);
  ast::Integer integer = parse_integer(cursor);
  EXPECT_EQ(integer.value(), 12345);
}

TEST(Parser, ParseSignature) {}

TEST(Parser, ParseBody) {}

TEST(Parser, EmptyClassEmbeddedComments) {
  std::string in = "empty \"a\" = \"a\"( \"a\" ) \"a\" ";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);

  EXPECT_STREQ(clazz.name().c_str(), "empty");
  EXPECT_EQ(clazz.locals().size(), 0);
  EXPECT_EQ(clazz.methods().size(), 0);
}

TEST(Parser, EmptyClassWithSuper) {
  std::string in = "EmptyWithSuper = MySuper ()";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);
  EXPECT_STREQ(clazz.name().c_str(), "EmptyWithSuper");
  EXPECT_TRUE(clazz.super());
  EXPECT_STREQ(clazz.super().value().c_str(), "MySuper");
  EXPECT_EQ(clazz.locals().size(), 0);
  EXPECT_EQ(clazz.methods().size(), 0);
}

TEST(Parser, SymbolKind) {
  EXPECT_EQ(ast::SymbolKind::OPERATOR, ast::symbol_kind(ast::Symbol("+")));
  EXPECT_EQ(ast::SymbolKind::OPERATOR, ast::symbol_kind(ast::Symbol("-")));
  EXPECT_EQ(ast::SymbolKind::IDENTIFIER, ast::symbol_kind(ast::Symbol("test")));
  EXPECT_EQ(ast::SymbolKind::KEYWORD, ast::symbol_kind(ast::Symbol("test:")));
}

TEST(Parser, SimpleUnarySignature) {
  std::string in = "xyz";
  ParseCursor cursor(in);
  auto s = parse_signature(cursor);
  ASSERT_EQ(s->sigkind(), ast::SignatureKind::UNARY);
  auto sig = ast::signature_cast<ast::UnarySignature>(s.get());
  EXPECT_STREQ(sig->symbol().c_str(), "xyz");
}

TEST(Parser, SimpleBinarySignature) {
  std::string in = "+ arg";
  ParseCursor cursor(in);
  auto s = parse_signature(cursor);
  ASSERT_EQ(s->sigkind(), ast::SignatureKind::BINARY);
  auto sig = ast::signature_cast<ast::BinarySignature>(s.get());
  EXPECT_STREQ(sig->symbol().c_str(), "+");
  EXPECT_STREQ(sig->argument().c_str(), "arg");
}

TEST(Parser, SingleKeywordSignature) {
  std::string in = "a: bbb";
  ParseCursor cursor(in);
  auto s = parse_signature(cursor);
  ASSERT_EQ(s->sigkind(), ast::SignatureKind::KEYWORD);
  auto sig = ast::signature_cast<ast::KeywordSignature>(s.get());

  EXPECT_TRUE(sig->arguments().size() == 1);

  auto arg = sig->arguments()[0];
  EXPECT_STREQ(arg.keyword.c_str(), "a:");
  EXPECT_STREQ(arg.argument.c_str(), "bbb");
}

TEST(Parser, DoubleKeywordSignature) {
  std::string in = "aa: bbb cccc: dddd";
  ParseCursor cursor(in);
  auto s = parse_signature(cursor);
  ASSERT_EQ(s->sigkind(), ast::SignatureKind::KEYWORD);
  auto sig = ast::signature_cast<ast::KeywordSignature>(s.get());

  EXPECT_TRUE(sig->arguments().size() == 2);

  auto arg = sig->arguments()[0];
  EXPECT_STREQ(arg.keyword.c_str(), "aa:");
  EXPECT_STREQ(arg.argument.c_str(), "bbb");

  arg = sig->arguments()[1];
  EXPECT_STREQ(arg.keyword.c_str(), "cccc:");
  EXPECT_STREQ(arg.argument.c_str(), "dddd");
}

TEST(Parser, TripleKeywordSignature) {
  std::string in = "a: b c: d e: f";
  ParseCursor cursor(in);
  auto s = parse_signature(cursor);
  ASSERT_EQ(s->sigkind(), ast::SignatureKind::KEYWORD);
  auto sig = ast::signature_cast<ast::KeywordSignature>(s.get());

  EXPECT_TRUE(sig->arguments().size() == 3);

  auto arg = sig->arguments()[0];
  EXPECT_STREQ(arg.keyword.c_str(), "a:");
  EXPECT_STREQ(arg.argument.c_str(), "b");

  arg = sig->arguments()[1];
  EXPECT_STREQ(arg.keyword.c_str(), "c:");
  EXPECT_STREQ(arg.argument.c_str(), "d");

  arg = sig->arguments()[2];
  EXPECT_STREQ(arg.keyword.c_str(), "e:");
  EXPECT_STREQ(arg.argument.c_str(), "f");
}

TEST(Parser, ParseUnaryExpression) {
  std::string in = "object message";
  ParseCursor cursor(in);
  auto e = parse_expression(cursor);
  ASSERT_EQ(e->exprkind(), ast::ExpressionKind::UNARY);
  auto expr = ast::expression_cast<ast::UnaryExpression>(e.get());
  // EXPECT_STREQ(expr->receiver().c_str(), "object");
  EXPECT_STREQ(expr->message().c_str(), "message");
}

TEST(Parser, ParseExpression1) { std::string in = "a b: c"; }

TEST(Parser, ParseExpression2) { std::string in = "a b c"; }

TEST(Parser, ParseExpression3) { std::string in = "a + b"; }

TEST(Parser, ParseUnaryMethod) {
  std::string in = "a = ( | b | ^c )";
  ParseCursor cursor(in);
  ast::Method method = parse_method(cursor);
}

TEST(Parser, ParseBinaryMethod) {
  std::string in = "+ arg = ( | b | ^c )";
  ParseCursor cursor(in);
  ast::Method method = parse_method(cursor);
}

TEST(Parser, ParseKeywordMethod) {
  std::string in = "a: a b: b = ( | c | ^a + b )";
  ParseCursor cursor(in);
  ast::Method method = parse_method(cursor);
}

TEST(Parser, EmptyClass) {
  std::string in = "EmptyClass = ( ) ";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);
  EXPECT_STREQ(clazz.name().c_str(), "EmptyClass");
  EXPECT_FALSE(clazz.super());
  EXPECT_EQ(clazz.locals().size(), 0);
  EXPECT_EQ(clazz.methods().size(), 0);
}

TEST(Parser, EmptyClassWithEmptyLocals) {
  std::string in = "EmptyClass = ( | | ) ";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);
  EXPECT_STREQ(clazz.name().c_str(), "EmptyClass");
  EXPECT_FALSE(clazz.super());
  EXPECT_EQ(clazz.locals().size(), 0);
  EXPECT_EQ(clazz.methods().size(), 0);
}

TEST(Parser, ClassWithFields) {
  std::string in = "ClassWithFields = ( | frog  frog2 |) ";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);
  EXPECT_STREQ(clazz.name().c_str(), "ClassWithFields");
  EXPECT_FALSE(clazz.super());
  EXPECT_EQ(clazz.locals().size(), 2);
  EXPECT_STREQ(clazz.locals()[0]->c_str(), "frog");
  EXPECT_STREQ(clazz.locals()[1]->c_str(), "frog2");
  EXPECT_EQ(clazz.methods().size(), 0);
}

// TEST(Parser, ClassWithUnaryMethod) {
//   std::string in = slurp(test::env->resource("ClassWithUnaryMethod.som"));
//   ParseCursor cursor(in);
//   ast::Clazz clazz = parse_clazz(cursor);
//   EXPECT_STREQ(clazz.name().c_str(), "ClassWithUnaryMethod");
//   ASSERT_EQ(clazz.methods().size(), 1);
//   auto sig = clazz.methods()[0]->signature();
//   ASSERT_EQ(sig->sigkind(), SignatureKind::UNARY);
//   auto unary = std::static_pointer_cast<ast::UnarySignature>(sig);
//   EXPECT_STREQ(unary->symbol().c_str(), "test");
// }

// TEST(Parse, ClassWithBinaryMethod) {
//   std::string in = slurp(test::env->resource("ClassWithBinaryMethod.som"));
//   ParseCursor cursor(in);
//   ast::Clazz clazz = parse_clazz(cursor);
//   EXPECT_STREQ(clazz.name().c_str(), "ClassWithBinaryMethod");
//   ASSERT_EQ(clazz.methods().size(), 1);
//   auto sig = clazz.methods()[0]->signature();
//   ASSERT_EQ(sig->sigkind(), SignatureKind::BINARY);
//   auto binary = std::static_pointer_cast<ast::BinarySignature>(sig);
//   EXPECT_STREQ(binary->symbol().c_str(), "%");
//   EXPECT_STREQ(binary->argument().c_str(), "argument");
// }

TEST(Parser, ClassWithMethods) {
  std::string in = "ClassWithMethods = ()";
  ParseCursor cursor(in);
  ast::Clazz clazz = parse_clazz(cursor);
  EXPECT_STREQ(clazz.name().c_str(), "ClassWithMethods");
  EXPECT_FALSE(clazz.super());
  EXPECT_EQ(clazz.locals().size(), 0);
  EXPECT_EQ(clazz.methods().size(), 0);
}

// TEST(Parser, LoadEmpty) {
//   ast::Root ast = parse_file(test::env->resource("Fib.som"));
//   EXPECT_TRUE(ast.empty());
// }
