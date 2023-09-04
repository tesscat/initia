#ifndef NET_HPP
#define NET_HPP

#include <cmath>
#include <cstdint>
#include "layer.hpp"

template<typename T, const uintmax_t Layers, const uintmax_t InpSize, const uintmax_t OutSize>
class Network {
public:
  InputLayer<T, InpSize>& inl;
  std::vector<Layer<T>*> layers;
  Network(InputLayer<T, InpSize>& inl_, std::vector<Layer<T>*> layers_) : inl(inl_), layers(layers_) {
    for (uintmax_t i = 0; i < Layers; i++) {
      layers[i]->idx = i;
    }
  };

  // we need to build a set of 'nudges' that we want
  // one for each weight + one for each bias
  // we might as well store it w the layers

  // Network() {
  //   layers[0] = new InternalLayer<T, ActivFns[0], Nodes[1], Nodes[0]>(inl);
  //   for (uintmax_t i = 1; i < Layers - 1; i++) {
  //     layers[i] = new InternalLayer<T, ActivFns[i], Nodes[i], Nodes[i - 1]>(layers[i - 1]);
  //   }
  // }

  void AccumulateBackprop(Vector<T, InpSize> inp, Vector<T, OutSize> expected) {
    // T cost = Cost(inp, expected);
    Vector<T, OutSize> out = Run(inp);
    // T deriv_cost = DerivBasicCost()
    // std::cout << "Cost:" << cost << '\n';
    // cost *= -1;
    // hoo boy
    // for every output node,
    for (uintmax_t output_idx = 0; output_idx < OutSize; output_idx++) {
      // how far off are we?
      T cost = BasicCost(out[output_idx], expected[output_idx]);
      T d_cost = DerivBasicCost(out[output_idx], expected[output_idx]);
      // for every layer (excluding input ofc)
      for (uintmax_t layer = 1; layer < Layers; layer++) {
        // for every node
        for (uintmax_t node = 0; node < layers[layer]->size; node++) {
          // for every previous node
          for (uintmax_t prev = 0; prev < layers[layer - 1]->size; prev++) {
            // ie for every weight
            // take the gradient

            T nudge = layers[Layers - 1]->node_deriv_wrt_prev_weight(layer, node, prev, output_idx);
            nudge *= d_cost;
            nudge *= -1;
            // add it on
            std::cout << layer << 'd' << d_cost << '\n';
            layers[layer]->add_weight_nudge(prev, node, nudge);
          }
          // do the bias stuff
          T nudge = layers[Layers - 1]->node_deriv_wrt_prev_bias(layer, node, output_idx);
          nudge *= d_cost;
          nudge *= -1;
          layers[layer]->add_bias_nudge(node, nudge);
        }
      }
    }
  }

  void ApplyNudges(T rate, uintmax_t iters) {
    for (uintmax_t layer = 0; layer < Layers - 1; layer++) {
      layers[layer]->ApplyWeightNudges(rate, iters);
      layers[layer]->ApplyBiasNudges(rate, iters);
    }
  }
  
  T BasicCost(T out, T expected) {
    return 0.5*std::pow((expected - out), 2);
  }
  T DerivBasicCost(T out, T expected) {
    return (expected - out);
  }

  Vector<T, OutSize> Run(Vector<T, InpSize> inp) {
    inl.set(inp);
    return vec_from<T, OutSize>(layers[Layers - 2]->fwd());
  }
};

#endif // !NET_HPP
