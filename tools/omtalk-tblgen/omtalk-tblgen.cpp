#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>
#include <optional>
#include <set>

using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;

namespace cl {
using namespace llvm::cl;
}

enum ActionType { PrintRecords, GenTypes };
static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(PrintRecords, "print-records",
                                 "Print all records to stdout (default)"),
                      clEnumValN(GenTypes, "gen-type-defs",
                                 "Write out all type definitions.")));

static cl::OptionCategory typeGenCat("Options for -gen-types-*");
static cl::opt<std::string>
    selectedUniverse("universe", llvm::cl::desc("The universe to gen for"),
                     llvm::cl::cat(typeGenCat), llvm::cl::CommaSeparated);

namespace omtalk {

// Wrapper class for a universe definition
class Universe {
public:
  explicit Universe(const llvm::Record *def) : def(def) {}

  StringRef getName() const { return def->getValueAsString("name"); }

  std::vector<Record *> getUnionUniverses() const {
    return def->getValueAsListOfDefs("union");
  }

  StringRef getCppNamespace() const {
    return def->getValueAsString("cppNamespace");
  }

  const llvm::Record *getDef() const { return def; }

  bool operator==(const Universe &other) const { return def == other.def; }

  bool operator<(const Universe &other) const {
    return getName() < other.getName();
  }

private:
  const llvm::Record *def;
};

class VariadicType;

// Wrapper class for a type record
class Type {
public:
  explicit Type(const llvm::Record *def) : def(def) {}

  bool isOptional() const { return def->isSubClassOf("Optional"); }

  bool isVariadic() const { return def->isSubClassOf("Variadic"); }

  bool isPrimitive() const { return def->isSubClassOf("PrimitiveType"); }

  bool isAggregate() const { return def->isSubClassOf("AggregateType"); }

  // VariadicType getAsVariadic() const { return VariadicType(def); }

  StringRef getName() const { return def->getValueAsString("name"); }

  StringRef getDescription() const {
    return def->getValueAsString("description");
  }

  Universe getUniverse() const {
    return Universe(def->getValueAsDef("universe"));
  }

  const llvm::Record *getDef() const { return def; }

private:
  const llvm::Record *def;
};

class AggregateType : Type {
public:
  explicit AggregateType(const llvm::Record *def) : Type(def) {}

  std::optional<Type> getParent() const {
    auto *parent = getDef()->getValueAsOptionalDef("parent");
    if (parent) {
      return Type(parent);
    }
    return std::nullopt;
  }
};

class VariadicType : Type {
public:
  explicit VariadicType(const llvm::Record *def) : Type(def) {}

  const Type getBaseType() const {
    return Type(getDef()->getValueAsDef("baseType"));
  }
};

static bool emitAggregateTypeDef() { return false; }

static bool emitTypeDef(const llvm::RecordKeeper &keeper, raw_ostream &os,
                        const Type type) {

  const char *typeInfoStart = R"(
// {0}
class {1})";

  const char *typeInfoEnd = R"( { )";

  os << llvm::formatv(typeInfoStart, type.getDescription(), type.getName());

  if (type.isAggregate()) {
    AggregateType a(type.getDef());
    auto parent = a.getParent();
    if (parent) {
      os << llvm::formatv(R"( : public {0})", parent->getName());
    }
  }

  os << typeInfoEnd;


  os << "};\n\n";
  return false;
}

static bool emitUniverseTypes(const llvm::RecordKeeper &records,
                              raw_ostream &os) {
  llvm::emitSourceFileHeader("Type Definitions", os);

  // Get the current universe
  const llvm::Record *universeDef = nullptr;
  {
    auto universes = records.getAllDerivedDefinitions("Universe");
    if (universes.empty()) {
      return true;
    }

    auto it = llvm::find_if(universes, [](const llvm::Record *def) {
      return Universe(def).getName() == selectedUniverse;
    });
    if (it == universes.end()) {
      return true;
    }
    universeDef = *it;
  }

  // Get all types
  auto typeDefs = records.getAllDerivedDefinitions("Type");
  if (typeDefs.empty()) {
    return true;
  }

  // Collect all unioned universes
  std::set<Universe> universes;
  std::vector<Universe> toProcess;
  universes.emplace(universeDef);
  toProcess.emplace_back(universeDef);
  while (!toProcess.empty()) {
    const auto universe = toProcess.back();
    toProcess.pop_back();
    for (const auto &def : universe.getUnionUniverses()) {
      Universe other(def);
      auto [_, inserted] = universes.insert(other);
      if (inserted) {
        toProcess.push_back(other);
      }
    }
  }

  // Emit each type that is in one of the universes
  for (const auto typeDef : typeDefs) {
    Type type(typeDef);
    if (universes.count(type.getUniverse())) {
      emitTypeDef(records, os, type);
    }
  }

  return false;
}

} // namespace omtalk

static bool omtalkTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case PrintRecords:
    os << records;
    break;
  case GenTypes:
    return omtalk::emitUniverseTypes(records, os);
    break;
  }
  return false;
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm::llvm_shutdown_obj Y;

  return TableGenMain(argv[0], &omtalkTableGenMain);
}
