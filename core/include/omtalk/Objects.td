#ifndef OMTALK_OBJECT
#define OMTALK_OBJECT

//===----------------------------------------------------------------------===//
// Base
//===----------------------------------------------------------------------===//

// Language base.  All types should belong to one language type.
class Language {
    string name = ?;
}

// Marker used to identify the member list for an object
def members;

class Type {
    // The name of the type
    string name = ?;
}

class Primitive : Type {

    // The size of the type in bytes
    int size = 4;

    // Whether this field should be traced by the garbage collector
    bit trace = 0;
}

// An aggregate of other types.  Can inherit fields from other types.
class AggregateType : Type {
    AggregateType super = ?;
    dag fields = (members);
}

// The super class of this struct.  Fields of the super class will be
// included in the beginning of this object with identical layout.
class Super<AggregateType s> {
    AggregateType super = s;
}

class Variadic<Type type> {
    Type baseType = type;
}

class LangType<Language lang> : AggregateType {
    Language language = lang;
}

def I64 : Primitive {
    let name = "i64";
}

def Ref : Primitive {
    let name = "ref";
    let trace = 1;
}

def Box : Primitive {
    let name = "box";
    let trace = 1;
}

//===----------------------------------------------------------------------===//
// Omtalk
//===----------------------------------------------------------------------===//

def Omtalk : Language {
    let name = "Omtalk";
}

class OmtalkType : LangType<Omtalk>;

def MetaObject : OmtalkType {
    let name = "MetaObject";
    let fields = (members Ref:$klass);
}

def Object : OmtalkType, Super<MetaObject> {
    let name = "Object";
    let fields = (members Variadic<Box>:$fields);
}

def Klass : OmtalkType, Super<MetaObject> {
    let name = "Klass";
}

def Symbol : OmtalkType, Super<MetaObject> {
    let name = "Symbol";
    let fields = (I64:$value);
}

#endif // OMTALK_OBJECT