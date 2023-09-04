#ifndef FNS_HPP
#define FNS_HPP

#include <algorithm>
#include <cmath>
namespace fns {
template<typename T>
T ReLu(T in) {
  return std::max(in, 0.0);
}
template<typename T>
T Deriv_ReLu(T in) {
  return in > 0 ? 1 : 0;
}
template<typename T>
T ATan(T in) {
  return std::atan(in);
}
template<typename T>
T Deriv_ATan(T in) {
  return 1/(1 + in*in);
}
template<typename T>
T Logistic(T in) {
  return 1/(1 + std::exp(-in));
}
}
#endif // !FNS_HPP
