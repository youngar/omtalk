#ifndef OMTALK_STACK_HPP_
#define OMTALK_STACK_HPP_

#include <cstdint>

namespace omtalk {

constexpr std::size_t DEFAULT_STACK_SIZE = 100;

class Stack {
 public:
  Stack() : Stack(DEFAULT_STACK_SIZE) {}

  Stack(std::size_t size) : _data(new std::uint8_t[size]), _size(size) {}

  Stack(Stack& other)
      : _data(new std::uint8_t[other._size]), _size(other._size) {
    for (std::size_t i = 0; i < _size; ++i) {
      _data[i] = other._data[i];
    }
  }

  Stack(Stack&& other) {
    _data = other._data;
    other._data = nullptr;
    _size = other._size;
    other._size = 0;
  }

  ~Stack() {
    delete[] _data;
    _size = 0;
    _data = nullptr;
  }

  std::uint8_t* data() const { return _data; }
  std::size_t size() const { return _size; }

 private:
  std::uint8_t* _data;
  std::size_t _size;
};

}  // namespace omtalk

#endif  // OMTALK_STACK_HPP_
