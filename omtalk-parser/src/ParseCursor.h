#ifndef OMTALK_PARSECURSOR_H_
#define OMTALK_PARSECURSOR_H_

#include <cstdint>
#include <omtalk/parser/Location.h>
#include <string>
#include <string_view>

namespace omtalk {
namespace parser {

class ParseCursor {
public:
  using difference_type = ptrdiff_t;
  using value_type = char;
  using pointer = char *;
  using reference = char &;
  using iterator_category = std::input_iterator_tag;

  ParseCursor(const std::string_view &in) : _location(), _in(in) {}

  ParseCursor(const std::string &filename, const std::string_view &in)
      : _location{1, 1, 0, filename}, _in(in) {}

  ParseCursor(const Location &location, const std::string_view &in)
      : _location(location), _in(in) {}

  ParseCursor(const ParseCursor &) = default;

  ParseCursor(ParseCursor &&) = default;

  ParseCursor &operator=(const ParseCursor &) = default;

  ParseCursor &operator=(ParseCursor &&) = default;

  char operator*() const { return get(); }

  ParseCursor &operator++() {
    char c = get();
    _location.offset += 1;
    if (c == '\n') {
      _location.column = 0;
      _location.line += 1;
    } else {
      _location.column++;
    }
    return *this;
  }

  ParseCursor operator++(int) {
    ParseCursor copy = *this;
    ++(*this);
    return copy;
  }

  bool operator==(const ParseCursor &rhs) const {
    return location().offset == rhs.location().offset;
  }

  bool operator!=(const ParseCursor &rhs) { return !(*this == rhs); }

  const Location &location() const { return _location; }

  const std::size_t offset() const { return location().offset; }

  bool end() const { return offset() == in_length(); }

  const std::string_view &in() const { return _in; }

  std::size_t in_length() const { return in().length(); }

  std::string_view::const_iterator in_iter() const {
    return in().substr(offset()).begin();
  }

  char get(std::size_t off = 0) const {
    if (end())
      throw std::exception();
    return _in[offset() + off];
  }

  void filename(std::string filename) { _location.filename = filename; }

private:
  std::string_view _in;
  Location _location;
};

} // namespace parser
} // namespace omtalk

#endif