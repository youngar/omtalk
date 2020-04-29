#include <iostream>
#include <omtalk/parser/AST.h>
#include <omtalk/parser/Parser.h>

using namespace omtalk;
using namespace omtalk::parser;

namespace {

class ASTDumper {
public:
  void dump(const ClassDecl &classDecl);

private:
  // Indent indent() {
  //     return Indent(indent_)
  // }
  // class Indent {

  // };
  unsigned indent = 0;
};
} // namespace

void ASTDumper::dump(const ClassDecl &classDecl) {
  std::cout << "dumping a class!\n";
}

void omtalk::parser::dump(const ClassDecl &classDecl) {
  ASTDumper dumper;
  dumper.dump(classDecl);
}
