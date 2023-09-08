#ifndef FNS_HPP
#define FNS_HPP

#include <Eigen/Eigen>
#include <functional>
template<typename T>
inline Eigen::VectorX<T> Vectorize(Eigen::VectorX<T> in, std::function<T(T)> fn) {
  Eigen::VectorX<T> out(in.rows());
  for (uint i = 0; i < in.rows(); i++) {
    out(i) = fn(in(i));
  }
  return out;
}

template<typename T>
T Sigmoid(T in) {
  return 1.0/(1.0 + std::exp((-in)));
}
template<typename T>
T SigmoidDeriv(T in) {
  T sig = Sigmoid(in);
  return sig*(1-sig);
}

#endif // !FNS_HPP
