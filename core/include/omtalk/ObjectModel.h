#ifndef OMTALK_OBJECTMODEL_H
#define OMTALK_OBJECTMODEL_H

#include <omtalk/Ref.h>
#include <omtalk/Tracing.h>

struct VMKlass;

class OmtalkObject;
class OmtalkValue;

using Slot = std::uintptr_t;

struct VMBase {
  
};

struct VMObject : VMBase {
  VMKlass *klass;
};

struct VMKlass : VMBase {
  VMKlass *klass;
};

struct VMSymbol :VMBase {
  VMKlass *klass;
};

class VMObjectProxy {
public:
  VMObjectProxy(void *target) : target(target) {}

  Slot &getHeader() { return getSlot(0); }

  Slot &getKlass() { return getSlot(1); }

  Slot &getField(unsigned field) { return getSlot(2 + field); }

  Slot &getSlot(unsigned slot) { return static_cast<Slot *>(target)[slot]; }

private:
  void *target;
};

class OmtalkObjectProxy {
public:
private:
};

class OmtalkSlotProxy {
public:
  OmtalkSlotProxy(Slot value) : value(value) {}
  Slot getValue() { return value; }

private:
  Slot value;
};

class OmtalkCollectorScheme {
  using ObjectProxy = OmtalkObjectProxy;
  using SlotProxy = OmtalkSlotProxy;
};

#endif