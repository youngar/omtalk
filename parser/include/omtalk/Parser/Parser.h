#ifndef OMTALK_PARSER_PARSER_H_
#define OMTALK_PARSER_PARSER_H_

#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>

namespace omtalk {
namespace parser {

std::unique_ptr<ClassDecl> parseClassFile(std::string filename);

void dump(const ClassDecl &classDecl);

// inline ast::Root parse(const std::string &filename,
//                        const std::string_view &in) {
//   ParseCursor cursor(filename, in);
//   return parse(cursor);
// }

// inline ast::Root parse(const std::string_view &in) { return parse("<in>",
// in); }

// inline std::string slurp(const std::string &filename) {
//   std::ifstream in(filename, std::ios::in);
//   std::stringstream buffer;
//   buffer << in.rdbuf();
//   return buffer.str();
// }

// inline ast::Root parse_file(const std::string &filename) {
//   std::string in = slurp(filename);
//   return parse(filename, in);
// }

} // namespace parser
} // namespace omtalk

#endif
