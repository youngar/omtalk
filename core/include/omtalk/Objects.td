#ifndef OMTALK_OBJECT
#define OMTALK_OBJECT

//===----------------------------------------------------------------------===//
// Base
//===----------------------------------------------------------------------===//

// A set of all possible types.
class Universe {
    // The name of the universe
    string name = ?;

    // A list of other universes which are included in this universe
    list<Universe> union = [];
}

class Union<list<Universe> universes> {
    list<Universe> union = universes;
}

// Marker used to identify the member list for an object
def members;

// Represents a type belonging to a universe
class Type<Universe u> {
    // The name of the type
    string name = ?;

    // One line description of the type
    string description = "";

    // The universe this type belongs to
    Universe universe = u;
}

// Represents a primitive type.
class PrimitiveType<Universe u> : Type<u> {

    // The size of the type in bytes
    int size = 4;

    // Whether this field should be traced by the garbage collector
    bit trace = 0;
}

// An aggregate of other types.  Can inherit fields from other types.
class AggregateType<Universe u> : Type<u> {
    AggregateType parent = ?;
    dag fields = (members);
}

// Represents single inheritance from a parent. The parent and child struct
// must both be aggregate types.  All parent fields are inlined before child
// fields with identical layout.
class Inherit<AggregateType s> {
    AggregateType parent = s;
}

// A type derived from another.  Belongs to the universe
// of the underlying type.
class Derived<Type type> {
    Universe universe = type.universe;
    Type baseType = type;
}

// An optional instance of a type
class Optional<Type type> : Derived<type>;

// A variable number of a type
class Variadic<Type type> : Derived<type>;

//===----------------------------------------------------------------------===//
// Core Types
//===----------------------------------------------------------------------===//

def Core : Universe {
    let name = "Core";
}

class CoreType : PrimitiveType<Core>;

def I32 : CoreType {
    let name = "i64";
    let description = "32 bit integer";
    let size = 4;
}

def I64 : CoreType {
    let name = "i64";
    let description = "64 bit integer";
    let size = 8;
}

def Ref : CoreType {
    let name = "ref";
    let description = "A reference";
    let size = 8;
    let trace = 1;
}

//===----------------------------------------------------------------------===//
// Omtalk
//===----------------------------------------------------------------------===//

def Omtalk : Universe, Union<[Core]> {
    let name = "Omtalk";
}

class OmtalkType : AggregateType<Omtalk>;

def Box : PrimitiveType<Omtalk> {
    let name = "box";
    let size = 8;
    let trace = 1;
}

def MetaObject : OmtalkType {
    let name = "MetaObject";
    let fields = (members Ref:$klass);
}

def Object : OmtalkType, Inherit<MetaObject> {
    let name = "Object";
    let fields = (members Variadic<Box>:$fields);
}

def Klass : OmtalkType, Inherit<MetaObject> {
    let name = "Klass";
    let fields = (members Variadic<Box>:$fields);
}

def Symbol : OmtalkType, Inherit<MetaObject> {
    let name = "Symbol";
    let fields = (I64:$value);
}

#endif // OMTALK_OBJECT