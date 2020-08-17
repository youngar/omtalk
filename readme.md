# Omtalk

Omtalk is a static compiler for [SOM], a derivative of smalltalk.
Omtalk is built with [MLIR] and [LLVM].

[SOM]:  http://som-st.github.io/
[MLIR]: https://mlir.llvm.org/
[LLVM]: https://llvm.org/

## Code Generation Example

Omtalk currently compiles SOM code to a high-level representation in MLIR.
Lowering to LLVM is WIP. The Omtalk runtime uses a single tagged value type,
called an `!omtalk.box<?>` in MLIR.

### Fibonacci

The following code:

```smalltalk
Fibonacci = (          "defines a subclass of Object"
    fib: n = (         "defines the fib method with the argument n"
        ^ n <= 1
            ifTrue:  1
            ifFalse: [ self fib: (n - 1) + (self fib: (n - 2)) ]
    )
)
```

Compiles to the MLIR code:

```mlir
module {
  omtalk.klass @Fibonacci : @Object {
    "omtalk.method"() ( {
    ^bb0(%arg0: !omtalk.box<?>):  // no predecessors
      %0 = "omtalk.constant_int"() {value = 1 : i64} : () -> !omtalk.box<int>
      %1 = "omtalk.send"(%arg0, %0) {message = @"<="} : (!omtalk.box<?>, !omtalk.box<int>) -> !omtalk.box<?>
      %2 = "omtalk.constant_int"() {value = 1 : i64} : () -> !omtalk.box<int>
      %3 = "omtalk.block"() ( {
        %5 = "omtalk.self"() : () -> !omtalk.box<ref>
        %6 = "omtalk.constant_int"() {value = 1 : i64} : () -> !omtalk.box<int>
        %7 = "omtalk.send"(%arg0, %6) {message = @"-"} : (!omtalk.box<?>, !omtalk.box<int>) -> !omtalk.box<?>
        %8 = "omtalk.self"() : () -> !omtalk.box<ref>
        %9 = "omtalk.constant_int"() {value = 2 : i64} : () -> !omtalk.box<int>
        %10 = "omtalk.send"(%arg0, %9) {message = @"-"} : (!omtalk.box<?>, !omtalk.box<int>) -> !omtalk.box<?>
        %11 = "omtalk.send"(%8, %10) {message = @"fib:"} : (!omtalk.box<ref>, !omtalk.box<?>) -> !omtalk.box<?>
        %12 = "omtalk.send"(%7, %11) {message = @"+"} : (!omtalk.box<?>, !omtalk.box<?>) -> !omtalk.box<?>
        %13 = "omtalk.send"(%5, %12) {message = @"fib:"} : (!omtalk.box<ref>, !omtalk.box<?>) -> !omtalk.box<?>
        "omtalk.return"(%13) : (!omtalk.box<?>) -> ()
      }) {type = () -> !omtalk.box<?>} : () -> !omtalk.box<?>
      %4 = "omtalk.send"(%1, %2, %3) {message = @"ifTrue:ifFalse:"} : (!omtalk.box<?>, !omtalk.box<int>, !omtalk.box<?>) -> !omtalk.box<?>
      "omtalk.return"(%4) : (!omtalk.box<?>) -> ()
    }) {sym_name = "fib:", type = (!omtalk.box<?>) -> !omtalk.box<?>} : () -> ()
  }
}
```

## Building

To build Omtalk and all tests, run:

```sh
mkdir build; cd build
cmake -G Ninja -C ../cmake/caches/dev.cmake ..
ninja
```
