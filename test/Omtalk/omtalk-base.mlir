// RUN: omtalk-opt %s | omtalk-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        return
    }
}

module {
  func @add_ten(%arg0: !omtalk.box<?>) -> !omtalk.box<?> {
    %0 = "omtalk.constant"() {value = 10 : i64} : () -> !omtalk.box<?>
    %1 = "omtalk.iadd"(%0, %arg0) : (!omtalk.box<?>, !omtalk.box<?>) -> !omtalk.box<?>
    "omtalk.return"(%1) : (!omtalk.box<?>) -> ()
  }
  func @run() -> i64 {
    %0 = "omtalk.constant"() {value = 5 : i64} : () -> i64
    %1 = "omtalk.static_send"(%0) {message = @add_ten} : (i64) -> i64
    "omtalk.return"(%1) : (i64) -> ()
  }
}
