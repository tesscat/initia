#include "fns.hpp"
#include "mtx.hpp"
#include "layer.hpp"
#include <iostream>


int main() {
  // Matrix<int, 3, 3> k(1);
  // k[0][0] = 1;
  // k[1][0] = 2;
  // k[0][1] = 3;
  // k[1][1] = 4;
  // k.print(std::cout);
  // std::cout << std::endl;
  // Matrix<int, 2, 3> m(1);
  // m[0][0] = 4;
  // m[1][0] = 3;
  // m[0][1] = 2;
  // m[1][1] = 1;
  // m.print(std::cout);
  // std::cout << std::endl;
  // auto j = k*m;
  // std::cout << std::endl;
  // j.print(std::cout);
  // std::cout << std::endl;

  InputLayer<int, 8> inl;
  InternalLayer<int, ReLu, 4, 8> l1(inl);

  inl.set(vec_from<int, 8>({1, 1, 1, 1, 1, 1, 1, 1}));
  l1.fwd().print(std::cout);
  return 0;
}
