#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
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

  explicit Type(const llvm::DefInit *init) : def(init->getDef()) {}

  bool isOptional() const { return def->isSubClassOf("Optional"); }

  bool isVariadic() const { return def->isSubClassOf("Variadic"); }

  bool isPrimitive() const { return def->isSubClassOf("PrimitiveType"); }

  bool isAggregate() const { return def->isSubClassOf("AggregateType"); }

  StringRef getName() const { return def->getValueAsString("name"); }

  StringRef getDescription() const {
    return def->getValueAsString("description");
  }

  Universe getUniverse() const {
    return Universe(def->getValueAsDef("universe"));
  }

  const llvm::Record *getDef() const { return def; }

  bool operator==(const Type &other) const { return def == other.def; }

  bool operator<(const Type &other) const {
    return getName() < other.getName();
  }

private:
  const llvm::Record *def;
};

class AggregateType : public Type {
public:
  explicit AggregateType(const llvm::Record *def) : Type(def) {}

  explicit AggregateType(const Type &type) : Type(type) {}

  std::optional<Type> getParent() const {
    auto *parent = getDef()->getValueAsOptionalDef("parent");
    if (parent) {
      return Type(parent);
    }
    return std::nullopt;
  }

  llvm::DagInit *getFields() const {
    return getDef()->getValueAsDag("fields");
    ;
  }

  // Type getFieldType(int index) const {
  //   llvm::DagInit *fields = getDef()->getValueAsDag("fields");
  //   return Type(llvm::cast<llvm::DefInit>(fields->getArg(index)));
  // }

  StringRef getFieldName(int index) const {
    llvm::DagInit *fields = getDef()->getValueAsDag("fields");
    return fields->getArgNameStr(index);
  }
};

class VariadicType : public Type {
public:
  explicit VariadicType(const llvm::Record *def) : Type(def) {}

  explicit VariadicType(const Type &type) : Type(type) {}

  const Type getBaseType() const {
    return Type(getDef()->getValueAsDef("baseType"));
  }
};

unsigned emitField() { return 0; }

static bool emitAggregateTypeDef(const llvm::RecordKeeper &keeper,
                                 raw_ostream &os, const AggregateType type) {

  const char *typeInfoStart = R"(
// {0}
class {1})";
  const char *typeInfoEnd = "{\npublic:\n";

  os << llvm::formatv(typeInfoStart, type.getDescription(), type.getName());
  auto parent = type.getParent();
  if (parent) {
    os << llvm::formatv(R"( : public {0} )", parent->getName());
  }
  os << typeInfoEnd;

  // Print the constructors
  const char *baseConstructor =
      "  explicit {0}(void *address) : address(address) ";
  const char *childConstructor =
      "  explicit {0}(void *address) : {1}(address) ";
  if (!parent) {
    os << llvm::formatv(baseConstructor, type.getName());
  } else {
    os << llvm::formatv(childConstructor, type.getName(), parent->getName());
  }
  os << "{ }\n";

  if (!parent) {
    os << "\n";
    os << "  void *getAddress() const { return address; }\n";
  }

  // Print the fields
  os << "/*\n";
  // auto fields = type.getFields();
  // for (const auto &f : fields->getArgNames()) {
  //   os << f;
  // }
  // fields->dump();
  // fields->print(os);
  os << "*/\n";

  if (!parent) {
    os << "private:\n";
    os << "  void *address;\n";
  }

  os << "};\n\n";

  return false;
}

static bool emitTypeDef(const llvm::RecordKeeper &keeper, raw_ostream &os,
                        const Type type) {
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
    return false;
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

  // Emit each type that is in one of the universes, respecting that parent
  // types must be emitted first.
  std::set<Type> types;
  for (const auto &typeDef : typeDefs) {
    // types.insert(Type(typeDef));
  }

  while (!types.empty()) {
    for (const auto &type : types) {
      if (type.isPrimitive()) {
        // skip primitive types for now
        types.erase(type);
        continue;
      }

      // Emit an aggregate type only if its parent type has been emitted
      if (type.isAggregate()) {
        AggregateType aggType(type);
        auto parent = aggType.getParent();
        if (parent && types.count(*parent)) {
          // Parent has not been emitted yet, skip the child
          continue;
        }
        emitAggregateTypeDef(records, os, aggType);
      }

      types.erase(type);
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
