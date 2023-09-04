#ifndef LAYER_HPP
#define LAYER_HPP

#include "mtx.hpp"
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

template<typename T>
class Layer {
public:
  uintmax_t idx;
  uintmax_t size;
  std::vector<T> cached;
  virtual std::vector<T> fwd() = 0;
  virtual void add_weight_nudge(uintmax_t prev_idx, uintmax_t my_idx, T nudge) = 0;
  virtual void add_bias_nudge(uintmax_t my_idx, T nudge) = 0;
  virtual T node_deriv_wrt_prev_weight(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t prev_idx, uintmax_t my_idx) = 0;
  virtual T node_deriv_wrt_prev_bias(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t my_idx) = 0;
  virtual void ApplyBiasNudges(T rate, uintmax_t iters) = 0; 
  virtual void ApplyWeightNudges(T rate, uintmax_t iters) = 0;
};

template<typename T>
using activfnr_t = T(T);
template<typename T>
using activfn_t = activfnr_t<T>*;

template<typename T, activfn_t<T> ActivFn, activfn_t<T> DerivFn, const uintmax_t S, const uintmax_t PrevS>
class InternalLayer : public Layer<T> {
public:
  Layer<T>& prev;
  Matrix<T, PrevS, S> weights;
  Matrix<T, PrevS, S> weight_nudges;
  Vector<T, S> biases;
  Vector<T, S> bias_nudges;
  Vector<T, S> unactiv_cached;
  InternalLayer(Layer<T> & p) : prev(p) {
    // TODO: change
    this->size = S;
    weights.fill(1);
    biases.fill(0);

    weight_nudges.fill(0);
    bias_nudges.fill(0);
  }

  void GenWeights(std::function<T()> fn) {
    weights.fill(fn);
  }
  void GenBiases(std::function<T()> fn) {
    biases.fill(fn);
  }

  std::vector<T> fwd() override {
    Vector<T, S> out = weights * prev.fwd();
    out += biases;
    unactiv_cached = out;
    out.foreach(ActivFn);
    this->cached = out.flatten();
    return this->cached;
  }
  
  void ApplyBiasNudges(T rate, uintmax_t iters) override {
    bias_nudges *= rate/iters;
    std::cout << "bias nudges:\n";
    bias_nudges.print(std::cout);
    biases += bias_nudges;
    bias_nudges *= 0;
  }

  void ApplyWeightNudges(T rate, uintmax_t iters) override {
    weight_nudges *= rate/iters;
    std::cout << "weight nudges:\n";
    weight_nudges.print(std::cout);
    weights += weight_nudges;
    weight_nudges *= 0;
  }

  // The derivative of a node wrt the node before it, ie the weight x activ fn
  T node_deriv_wrt_imm_prev_node(uintmax_t prev_idx, uintmax_t my_idx) {
    return weights[prev_idx][my_idx] * DerivFn(unactiv_cached[my_idx]); 
  }

  T node_deriv_wrt_my_bias(uintmax_t my_idx) {
    return DerivFn(unactiv_cached[my_idx]);
  }

  // The derivative of a node wrt to a specific weight, ie the previous activation
  T node_deriv_wrt_imm_weight(uintmax_t prev_idx, uintmax_t my_idx) {
    T deriv = DerivFn(unactiv_cached[my_idx]);
    T q = prev.cached[prev_idx] * deriv;
    std::cout << "NDWIW" << this->idx << ' ' << q << '\n';
    return q;
  }

T node_deriv_wrt_prev_weight(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t prev_idx, uintmax_t my_idx) override {
    if (layer_idx == this->idx) {
      if (my_idx == node_idx)
        return node_deriv_wrt_imm_weight(prev_idx, my_idx);
      else
        return 0;
    }
    // the sum of all the inbound nodes wrt the target * the deriv of me wrt the inbound
    T sum = 0;
    for (uintmax_t i = 0; i < prev.size; i++) {
      sum += node_deriv_wrt_imm_prev_node(i, my_idx) * prev.node_deriv_wrt_prev_weight(layer_idx, node_idx, prev_idx, i);
    }

    return sum;
  }

  T node_deriv_wrt_prev_bias(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t my_idx) override {
    if (layer_idx == this->idx) {
      if (my_idx == node_idx)
        return node_deriv_wrt_my_bias(node_idx);
      else
        return 0;
    }
    // the sum of all the inbound nodes wrt the target * the deriv of me wrt the inbound
    T sum = 0;
    for (uintmax_t i = 0; i < prev.size; i++) {
      sum += node_deriv_wrt_imm_prev_node(i, my_idx) * prev.node_deriv_wrt_prev_bias(layer_idx, node_idx, i);
    }

    return sum;
  }
  void add_weight_nudge(uintmax_t prev_idx, uintmax_t my_idx, T nudge) override {
    // std::cout << "addind weight nudge of " << nudge << "\n";
    weight_nudges[prev_idx][my_idx] += nudge;
  }
  void add_bias_nudge(uintmax_t my_idx, T nudge) override {
    // std::cout << "addind bias nudge of " << nudge << "\n";
    bias_nudges[my_idx] += nudge;
  }
};

template<typename T, const uintmax_t S>
class InputLayer : public Layer<T> {
  Vector<T, S> data;
public:
  InputLayer() {
    this->size = S;
    this->cached = std::vector<T>(S);
  };
  void set(Vector<T, S> q) {
    data = q;
    this->cached = q.flatten();
  };
  std::vector<T> fwd() override {
    this->cached = data.flatten();
    return this->cached;
  }
  void add_weight_nudge(uintmax_t prev_idx, uintmax_t my_idx, T nudge) override {};
  void add_bias_nudge(uintmax_t my_idx, T nudge) override {};
  T node_deriv_wrt_prev_weight(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t prev_idx, uintmax_t my_idx) override {
    exit(2);
    return 1;
  };
  T node_deriv_wrt_prev_bias(uintmax_t layer_idx, uintmax_t node_idx, uintmax_t my_idx) override {exit(3);return 0;};
  void ApplyBiasNudges(T rate, uintmax_t iters) override {}; 
  void ApplyWeightNudges(T rate, uintmax_t iters) override {};
};

#endif
