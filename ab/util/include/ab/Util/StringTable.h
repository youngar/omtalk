#ifndef AB_UTIL_STRINGTABLE_H
#define AB_UTIL_STRINGTABLE_H

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

class Hash;

namespace ab {

struct StringRef {
public:
private:
  std::string str;
};

struct CachedString {
  StringRef string;
  Hash hash;
};

class Symbol {};

class StringTable {
public:
  const char *operator[](const std::string &s);

  const char *operator[](std::string &&s);

  const char *operator[](const char *s);

private:
  std::unordered_set<std::string> _strings;
};

inline const char *StringTable::operator[](const std::string &s) {
  auto it = _strings.insert(s).first;
  return (*it).c_str();
}

inline const char *StringTable::operator[](std::string &&s) {
  auto it = _strings.insert(std::move(s)).first;
  return (*it).c_str();
}

inline const char *StringTable::operator[](const char *s) {
  auto it = _strings.emplace(s).first;
  return (*it).c_str();
}

} // namespace ab

#endif // AB_UTIL_STRINGTABLE_H