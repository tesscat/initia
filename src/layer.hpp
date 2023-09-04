#ifndef LAYER_HPP
#define LAYER_HPP

#include "mtx.hpp"

template<typename T, const uintmax_t S>
class Layer {
public:
  const uintmax_t size = S;
  virtual Vector<T, S> fwd() = 0;
};

template<typename T>
using activfn_t = T(T);

template<typename T, activfn_t<T> ActivFn, const uintmax_t S, const uintmax_t PrevS>
class InternalLayer : public Layer<T, S> {
  Layer<T, PrevS>& prev;
  Matrix<T, PrevS, S> weights;
  Vector<T, S> biases;
public:
  InternalLayer(Layer<T, PrevS> & p) : prev(p) {
    // TODO: change
    weights.fill(1);
  }
  Vector<T, S> fwd() override {
    Vector<T, S> out = weights * prev.fwd();
    out += biases;
    out.foreach(ActivFn);
    return out;
  }
};

template<typename T, const uintmax_t S>
class InputLayer : public Layer<T, S> {
  Vector<T, S> data;
public:
  InputLayer() {};
  void set(Vector<T, S> q) {data = q;};
  Vector<T, S> fwd() override {return data;};
};

#endif
