#include <memory>
#include <omtalk/Parser/Parser.h>

using namespace omtalk;
using namespace omtalk::parser;

namespace {

class Parser final {
public:
  Parser(std::string filename) {}
  ~Parser(){};
  std::unique_ptr<ClassDecl> parseClass();

private:
};

} // namespace

std::unique_ptr<ClassDecl> Parser::parseClass() {
  UnknownLoc loc;
  std::string name = "test";
  auto classDecl = std::make_unique<ClassDecl>(loc, name);

  return classDecl;
}

std::unique_ptr<ClassDecl>
omtalk::parser::parseClassFile(std::string filename) {
  Parser parser(filename);
  return parser.parseClass();
}
