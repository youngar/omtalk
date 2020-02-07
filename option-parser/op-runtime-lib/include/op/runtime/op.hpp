#ifndef OP_RUNTIME_OP_HPP_
#define OP_RUNTIME_OP_HPP_

namespace op {

class Option {
  bool is_set = false;

 public:
  //   bool is_set() { return is_set; }
};

class BooleanOption {};

class ValueOption {};

class OptionSet {};

}  // namespace op

#endif  // OP_RUNTIME_OP_HPP_
