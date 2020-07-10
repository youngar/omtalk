#ifndef OMTALK_OM_OBJECTMODEL_H_
#define OMTALK_OM_OBJECTMODEL_H_

#include <cstdint>
#include <vector>

namespace omtalk {

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

using TypeTag = std::uint8_t;

namespace CoreTypes {
enum Types {
  /// integer types
  I8, I16, I32, I64,
  /// float types
  F32, F64,
  /// special types
  REF, PLY,
  // Marker for last type
  LAST_TYPE
};
}

class Type {
  virtual std::size_t getSize();
  virtual std::vector<std::size_t> &getOffsets();
  virtual std::vector<TypeTag> &getSlotDescription();
};

///
/// TODO: We can support automatically generated typetags if we support
/// uniqueing of types
///

template <typename BaseType>
class BasicType : public BaseType {
  virtual TypeTag getTag();

  std::vector<TypeTag> getSlotDescription() override {
    return { getTag(); }
  }

  std::vector<std::size_t> getOffsets() override { return {getSize()}; }
};

//===----------------------------------------------------------------------===//
// StructTypes
//===----------------------------------------------------------------------===//

template <typename Type>
class StructType {
  StructType(std::vector<Type>);

private:
  std::vector<Type> types;
};

class StructTypeBuilder {};

//===----------------------------------------------------------------------===//
// Omtalk Types
//===----------------------------------------------------------------------===//

class ShapeData;

class ShapeDataRef {};

extern ShapeDataRef SHAPE_DATA_REF;

namespace OmtalkTypes {
enum Types : TypeTag { ShapeDataRef, PolyVal, MyStruct };
}


struct MyStruct {
  void * my_thing;
};

// class MyStructType : public OmtalkType {
//   std::vector<std::byte> getTraceMap() override {
//     return {true};
//   }
//   std::size_t getSize() override { return sizeof(MyStruct); }

//   TypeTag getTag() override { return OmtalkTypes::MyStruct; }
// };

class OmtalkType : Type {
  virtual std::vector<std::byte> getTraceMap();
};

class OmtalkBasicType : public BasicType<OmtalkType> {
  virtual bool shouldTrace();

  std::vector<std::byte> getTraceMap() override {
    return {0};
  }
};

class ShapeDataRefType : public OmtalkBasicType {
  bool shouldTrace() override { return false; }
  TypeTag getTag() override { return OmtalkTypes::ShapeDataRef; }
  std::size_t getSize() override { return sizeof(ShapeData *); }
};

class PolyValType : public OmtalkBasicType {
  bool shouldTrace() override { return true; }
  TypeTag getTag() override { return OmtalkTypes::ShapeDataRef; }
  std::size_t getSize() override { return sizeof(void *); }
};

//===----------------------------------------------------------------------===//
// Type Base
//===----------------------------------------------------------------------===//

enum class SlotName : uintptr_t {};

//===----------------------------------------------------------------------===//
// Omtalk Types
//===----------------------------------------------------------------------===//

constexpr std::size_t CORE_TYPE_COUNT = 8;

class SlotInfo {
  Type* type;
  SlotName name;
};

using SlotMap = std::vector<bool>;

struct SlotTable {
  static std::size_t allocSize(std::size_t nslots) {
    return sizeof(SlotTable) + (sizeof(SlotInfo) * nslots);
  }
  SlotMap map;
  SlotInfo info[];
};

class ShapeInfo {
public:
  SlotTable slotTable;
};

enum class Symbol : std::uintptr_t {};

enum class Types : std::uint8_t {};

class SlotInfo {};

class KlassDescription {
public:
private:
};

class KlassData {
  Object *getKlassObject() { return klassObject; }
  std::size_t getInstanceSize() { return instanceSize; }

private:
  Object *klassObject;
  std::size_t instanceSize;
};

class Object {};

class ObjectHeader {
  Object *klass;
};

class KlassHeader {
  Object *klass;
};

class MetaKlass {};

} // namespace omtalk

#endif
