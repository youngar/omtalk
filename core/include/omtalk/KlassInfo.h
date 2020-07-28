#ifndef OMTALK_KLASSINFO_H
#define OMTALK_KLASSINFO_H

#include <string>
#include <vector>

namespace omtalk {

struct KlassInfo {
  std::string name;
  KlassInfo *super;
  std::vector<std::string> fields;
};

} // namespace omtalk

#endif