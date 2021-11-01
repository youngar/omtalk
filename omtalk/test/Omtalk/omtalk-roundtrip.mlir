// RUN: omtalk-opt %s | omtalk-opt | FileCheck %s

// CHECK: module
module {
  // CHECK: omtalk.klass @EmptyClass : @Object
  omtalk.klass @EmptyClass : @Object {
  }

  // CHECK: omtalk.klass @ClassWithSuper : @TheSuper
  omtalk.klass @ClassWithSuper : @TheSuper {
  }

  // CHECK: omtalk.klass @SmallClass : @Object
  omtalk.klass @SmallClass : @Object {
    // CHECK: "omtalk.field"() {sym_name = "A"} : () -> ()
    "omtalk.field"() {sym_name = "A"} : () -> ()
    // CHECK: "omtalk.field"() {sym_name = "B"} : () -> ()
    "omtalk.field"() {sym_name = "B"} : () -> ()
    // CHECK: "omtalk.field"() {sym_name = "C"} : () -> ()
    "omtalk.field"() {sym_name = "C"} : () -> ()
  }
}
