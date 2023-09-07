#ifndef FNS_HPP
#define FNS_HPP

#include <Eigen/Eigen>
#include <functional>
template<typename T, std::function<T(T)> fn>
Eigen::VectorX<T> Vectorize(Eigen::VectorX<T> in) {
  Eigen::VectorX<T> out(in.rows());
  for (uint i = 0; i < in.rows(); i++) {
    out(i) = fn(in(i));
  }
  return out;
}

#endif // !FNS_HPP
