#ifndef LAYER_HPP
#define LAYER_HPP

#include "mtx.hpp"
#include <vector>

template<typename T>
class Layer {
public:
  virtual std::vector<T> fwd() = 0;
};

template<typename T>
using activfn_t = T(T);

template<typename T, activfn_t<T> ActivFn, const uintmax_t S, const uintmax_t PrevS>
class InternalLayer : public Layer<T> {
  Layer<T>& prev;
  Matrix<T, PrevS, S> weights;
  Vector<T, S> biases;
public:
  InternalLayer(Layer<T> & p) : prev(p) {
    // TODO: change
    weights.fill(1);
  }
  std::vector<T> fwd() override {
    Vector<T, S> out = weights * prev.fwd();
    out += biases;
    out.foreach(ActivFn);
    return out.flatten();
    // return std::vector<T>();
  }
};

template<typename T, const uintmax_t S>
class InputLayer : public Layer<T> {
  Vector<T, S> data;
public:
  InputLayer() {};
  void set(Vector<T, S> q) {data = q;};
  std::vector<T> fwd() override {return data.flatten();};
};

#endif
