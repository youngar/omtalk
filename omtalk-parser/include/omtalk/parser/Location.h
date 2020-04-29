#ifndef OMTALK_PARSER_LOCATION_H_
#define OMTALK_PARSER_LOCATION_H_

#include <cstdint>
#include <memory>
#include <string>

namespace omtalk {
namespace parser {

class Location {
public:
  enum LocationKind {
    Loc_Unknown,
    Loc_LineCol,
    Loc_FileLineCol,
    Loc_Range,
  };

  Location(LocationKind kind) : kind(kind) {}
  virtual ~Location() = default;
  LocationKind getKind() const { return kind; }

private:
  const LocationKind kind;
};

class UnknownLoc : public Location {
public:
  UnknownLoc() : Location(Loc_Unknown) {}
  static bool classof(const Location *c) { return c->getKind() == Loc_Unknown; }
};

class LineColLoc : public Location {
public:
  LineColLoc(unsigned line, unsigned col)
      : Location(Loc_LineCol), line(line), col(col) {}

  static bool classof(const Location *c) { return c->getKind() == Loc_LineCol; }

private:
  unsigned line;
  unsigned col;
};

class FileLineColLoc : public Location {
public:
  FileLineColLoc(unsigned line, unsigned col,
                 std::shared_ptr<std::string> filename)
      : Location(Loc_FileLineCol), line(line), col(col), filename(filename){};
  static bool classof(const Location *c) {
    return c->getKind() == Loc_FileLineCol;
  }

private:
  unsigned line;
  unsigned col;
  std::shared_ptr<std::string> filename;
};

class RangeLoc : public Location {
public:
  RangeLoc(LineColLoc start, LineColLoc end, std::shared_ptr<std::string> filename)
      : Location(Loc_Range), start(start), end(end), filename(filename) {}

private:
  LineColLoc start;
  LineColLoc end;
  std::shared_ptr<std::string> filename;
};

} // namespace parser
} // namespace omtalk
#endif