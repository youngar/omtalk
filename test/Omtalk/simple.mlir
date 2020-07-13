// RUN: omtalk-opt %s | omtalk-opt | FileCheck %s

module {
  omtalk.klass @EmptyClass : @Object {
  }
  omtalk.klass @ClassWithSuper : @TheSuper {
  }
  omtalk.klass @SmallClass : @Object {
  }
}
