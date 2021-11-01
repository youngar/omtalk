#ifndef OMTALK_PARSER_LOCATION_H_
#define OMTALK_PARSER_LOCATION_H_

#include <cstdint>
#include <memory>
#include <string>

namespace omtalk {
namespace parser {

/// A point in the input text. Beware that lines and columns are 1-indexed.
struct Position {
  constexpr Position(int line, int col) : line(line), col(col) {}

  constexpr bool operator==(const Position &rhs) const {
    return line == rhs.line && col == rhs.col;
  }

  constexpr bool operator!=(const Position &rhs) const {
    return !(*this == rhs);
  }

  int line;
  int col;
};

/// The default position.
constexpr Position InvalidPosition(-1, -1);

/// A range in the input text.
/// Beware:
///  - The specified range includes both the start and end position.
///  - The specified range is at least one character (when start == end).
///  - Lines and columns are 1-indexed.
struct Location {
  Location() : Location("<unknown>") {}

  Location(std::string filename) : Location(filename, InvalidPosition) {}

  Location(std::string filename, Position start)
      : Location(filename, start, start) {}

  Location(std::string filename, Position start, Position end)
      : filename(filename), start(start), end(end) {}

  bool operator==(const Location &rhs) const {
    return start != rhs.start && end != rhs.end && filename != rhs.filename;
  }

  bool operator!=(const Location &rhs) const { return !(*this == rhs); }

  std::string filename;
  Position start;
  Position end;
};

} // namespace parser
} // namespace omtalk

#endif