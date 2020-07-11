#ifndef OMTALK_PARSECURSOR_H_
#define OMTALK_PARSECURSOR_H_

#include <cstdint>
#include <omtalk/Parser/Location.h>
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

  ParseCursor(const std::string_view &in)
      : filename(""), offset(0), position(1, 1), in(in) {}

  ParseCursor(const std::string &filename, const std::string_view &in)
      : filename(filename), offset(0), position(1, 1), in(in) {}

  ParseCursor(const ParseCursor &) = default;

  ParseCursor(ParseCursor &&) = default;

  ParseCursor &operator=(const ParseCursor &) = default;

  ParseCursor &operator=(ParseCursor &&) = default;

  char operator*() const { return get(); }

  ParseCursor &operator++() {
    char c = get();
    offset += 1;
    if (c == '\n') {
      position.col = 0;
      position.line += 1;
    } else {
      position.col++;
    }
    return *this;
  }

  ParseCursor operator++(int) {
    ParseCursor copy = *this;
    ++(*this);
    return copy;
  }

  bool operator==(const ParseCursor &rhs) const { return offset == rhs.offset; }

  bool operator!=(const ParseCursor &rhs) { return !(*this == rhs); }

  bool more() const { return !atEnd(); }

  bool atEnd() const { return offset == in.length(); }

  Position pos() const { return position; }

  Location loc() const { return {filename, position}; }

  const std::size_t getOffset() const { return offset; }

  // const std::string_view &in() const { return in; }

  // std::size_t in_length() const { return in().length(); }

  // std::string_view::const_iterator in_iter() const {
  //   return in().substr(offset).begin();
  // }

  char get(std::size_t off = 0) const {
    if (atEnd())
      throw std::exception();
    return in[offset + off];
  }

  const std::string &getFilename() const { return filename; }

  std::string subStringFrom(std::size_t start) const {
    return std::string(in.begin() + start, in.begin() + offset);
  }

private:
  std::string filename;
  std::string_view in;
  std::size_t offset;
  Position position;
};

} // namespace parser
} // namespace omtalk

#endif