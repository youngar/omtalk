#ifndef OMTALK_SYMBOLTABLE_HP_
#define OMTALK_SYMBOLTABLE_HP_

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace omtalk {

using Symbol = uintptr_t;

static const Symbol invalid_symbol = 0x0;

class SymbolTable {
public:
  class Iterator {
  public:
    Iterator &operator++() {
      ++_iter;
      return *this;
    }

    Symbol &operator*() const { return _iter->second; }

  private:
    friend class SymbolTable;

    explicit Iterator(
        const std::unordered_map<std::string, Symbol>::iterator &iter)
        : _iter(iter) {}

    std::unordered_map<std::string, Symbol>::iterator _iter;
  };

  bool contains(const std::string &s) { return _symbols.count(s) != 0; }

  bool contains(Symbol s) { return _strings.count(s) != 0; }

  Symbol intern(const std::string &s) {
    auto insertion = _symbols.insert({s, _next});

    if (std::get<bool>(insertion)) {
      assert(std::get<bool>(_strings.insert({_next, s})));
      ++_next;
    }

    return insertion.first->second;
  }

  std::string to_string(const Symbol &s) {
    assert(contains(s));
    return _strings[s];
  }

  Symbol operator[](const std::string &s) { return intern(s); }

  Iterator begin() { return Iterator(_symbols.begin()); }

  Iterator end() { return Iterator(_symbols.end()); }

private:
  Symbol _next = 1;
  std::unordered_map<Symbol, std::string> _strings;
  std::unordered_map<std::string, Symbol> _symbols;
};

class StringTable {
public:
  const char *operator[](const std::string &s) {
    auto it = _strings.insert(s).first;
    return (*it).c_str();
  }

  const char *operator[](std::string &&s) {
    auto it = _strings.insert(std::move(s)).first;
    return (*it).c_str();
  }

  const char *operator[](const char *s) {
    auto it = _strings.emplace(s).first;
    return (*it).c_str();
  }

private:
  std::unordered_set<std::string> _strings;
};

} // namespace omtalk

#endif // OMTALK_SYMBOL_HPP_
