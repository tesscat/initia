#ifndef FNS_HPP
#define FNS_HPP

#include "mtx.hpp"
#include <algorithm>
#include <cmath>
namespace fns {

template<typename T>
T Sigmoid(T in) {
  return 1/(1 + std::exp(in));
}
template<typename T>
Vector<T> Sigmoid(Vector<T> in) {
  Vector<T> out(in.height);
  for (uintmax_t i = 0; i < in.height; i++) {
    out[i] = Sigmoid(in[i]);
  }

  return out;
}

template<typename T>
T SigmoidPrime(T in) {
  return Sigmoid(in)/(1-Sigmoid(in));
}
template<typename T>
Vector<T> SigmoidPrime(Vector<T> in) {
  Vector<T> out(in.height);
  for (uintmax_t i = 0; i < in.height; i++) {
    out[i] = SigmoidPrime(in[i]);
  }

  return out;
}
}
#endif // !FNS_HPP
