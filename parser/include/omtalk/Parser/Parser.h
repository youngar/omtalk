#ifndef OMTALK_PARSER_PARSER_H_
#define OMTALK_PARSER_PARSER_H_

#include <omtalk/Parser/AST.h>
#include <omtalk/Parser/Location.h>

namespace omtalk {
namespace parser {

ModulePtr parseFile(const std::string &filename);

} // namespace parser
} // namespace omtalk

#endif
