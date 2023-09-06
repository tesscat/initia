#ifndef FNS_HPP
#define FNS_HPP

#include "mtx.hpp"
#include <algorithm>
#include <cmath>
namespace fns {

template<typename T>
T Sigmoid(T in) {
  return 1.0/(1.0 + std::exp(-in));
}
template<typename T>
Vector<T> Sigmoid(Vector<T> in) {
  Vector<T> out(in.height);
  for (uintmax_t i = 0; i < in.height; i++) {
    out.data[i][0] = Sigmoid(in.data[i][0]);
  }

  return out;
}

template<typename T>
T SigmoidPrime(T in) {
  T sg = Sigmoid(in);
  return sg*(1.0-sg);
}
template<typename T>
Vector<T> SigmoidPrime(Vector<T> in) {
  Vector<T> out(in.height);
  for (uintmax_t i = 0; i < in.height; i++) {
    out.data[i][0] = SigmoidPrime(in.data[i][0]);
  }

  return out;
}
}

#include<random>
namespace fns::tests {
void sigm() {
  std::mt19937_64 rng = std::mt19937_64();
  std::vector<double> v;
  for (int i = 0; i < 10; i++)
    v.push_back((double)(rng()/(rng.max()/10)));
  Vector<double> v1(v);

  v1.print(std::cout);
  std::cout << "sigm\n";
  Sigmoid(v1).print(std::cout);
}
};

#endif // !FNS_HPP
