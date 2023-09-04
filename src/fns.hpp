#ifndef FNS_HPP
#define FNS_HPP

#include <algorithm>
template<typename T>
T ReLu(T in) {
  return std::max(in, 0);
}

#endif // !FNS_HPP
