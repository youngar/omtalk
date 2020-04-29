// RUN: omtalk-opt %s | omtalk-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        return
    }
}
