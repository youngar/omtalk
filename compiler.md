# Mlir

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

# Base Syntax


(dialect/op-name #(in1 in2) #(out) (region-thing))

(core/module
    (core/func (arg0)
        (let %0 (constant 10)
            %1 (+ %arg0 %0)
            (return %1))))

(core/module 
    (core/func (do-thing (: %a0 i64) (: %a1 i64)) : i64
        (:= %0 (constant 10))
        (:= %1 (+ %arg0 %0))
        (return %1)))

(core/module
    (core/func
        (return (+ (core/i64 10) arg0)))))

# Extended Syntax

- imports
- lang declartions
- using statements / aliases / namespace resolution
- let / named assignment
- tree expressions: (+ (- 10 9) 2)

(namespace xxx)

# ??? Dialect

# Core Dialect
Implements the core dialect.  Responsible for defining the most basic operations that the reset of the language is built upon.  It defines the building blocks 

(namespace core)



# Standard Dialect

## Functions

(namespace std)
(comp/operation function)

;; 
(comp/protocol data function
    (name string))

## Integers

(comp/type i64)

(comp/operation + )

(comp/protocol data +
    )
(comp/protocol fold +
    ;; somehow fold plus operations
    (and (const @lhs)
         (const @rhs)
    (replace-all (+ @rhs @lhs))))


# Kaleidoscope Dialect

(namespace kal)

(comp/type-alias `kal )


(std/function ))

# Common Lisp

()

```
(using namespace cl)

(comp/type list (??))

(comp/type-alias cl/any (any-of i64 list))

(comp/operation cl/defun
    {:args (cl/any)
     :outs (cl/any))

(core/def-operator add)
(core/implement-protocol control-flow add
  (arg (type i64) (type i64))
  (res (type i64)))

(func run ()
  (let (o1 o2 o3) (dialect/node-name in1 in2))
    o2)


```

```
(cl/defun thingy (arg)
  (return arg)
```

## Extended Syntax



Node Definition
===============

```
(def-node <name>)
```


# Base IR


# IR