#ifndef NET_HPP
#define NET_HPP

#include <cstdint>
#include "layer.hpp"

template<typename T, const uintmax_t Layers, const uintmax_t Nodes[Layers]>
class Network {
  Layer* layers[Layers];
};

#endif // !NET_HPP
