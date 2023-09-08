#ifndef NET_HPP
#define NET_HPP

#include "fns.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

template<typename Scalar>
class Network {
public:
  using Matrix = Eigen::MatrixX <Scalar>;
  using Vector = Eigen::VectorX <Scalar>;
  using Operator = std::function<Scalar(Scalar)>;
private:
  std::vector<Vector> activations;
  std::vector<Vector> weighted_inps;

  std::vector<Matrix> weights;
  std::vector<Vector> biases;
  std::vector<Matrix> delta_weights;
  std::vector<Vector> delta_biases;
  std::vector<Vector> deltas;
  Scalar learningRate;
  uint layers;
  Operator fn;
  Operator deriv_fn;
public:
  Network(std::vector<uint> topology, Scalar rate, Operator fn_, Operator deriv_fn_) : learningRate(rate), layers(topology.size()), fn(fn_), deriv_fn(deriv_fn_) {
    activations.push_back(Vector(topology[0]));
    for (int i = 1; i < topology.size(); i++) {
      weights.push_back(Matrix::Zero(topology[i], topology[i - 1]));
      biases.push_back(Vector::Zero(topology[i]));

      deltas.push_back(Vector::Zero(topology[i]));
      delta_weights.push_back(Matrix::Zero(topology[i], topology[i - 1]));
      delta_biases.push_back(Vector::Zero(topology[i]));

      activations.push_back(Vector::Zero(topology[i]));
      weighted_inps.push_back(Vector::Zero(topology[i]));
    }

    std::mt19937_64 rng = std::mt19937_64();
    // chosen by dice roll, guaranteed to be random
    rng.seed(5);
    std::uniform_real_distribution<double> dis(-0.4, 0.4);  

    // randomize weights + biases
    for (int k = 0; k < topology.size() - 1; k++) {
      Matrix& v = weights[k];
      for (uint i = 0; i < v.rows(); i++) {
        for (uint j = 0; j < v.cols(); j++) {
          v(i, j) = dis(rng);
        }
      }
      Vector& b = biases[k];
      for (uint i = 0; i < b.rows(); i++) {
        b(i) = dis(rng);
      }
    }
  }
  Vector forward(Vector inp) {
    Vector activ = inp;
    activations[0] = activ;
    for (uint i = 0; i < layers - 1; i++) {
      weighted_inps[i] = weights[i] * activations[i] + biases[i];
      activations[i+1] = Vectorize<Scalar>(weighted_inps[i], fn);
    }

    return activations.back();
  }

  int onehotdecode(Vector v) {
    Scalar max = -std::numeric_limits<Scalar>::infinity();
    int index = -1;
    for (int i = 0; i < v.rows(); i++) {
      if (v(i) > max) {
        index = i;
        max = v(i);
      }
    }

    return index;
  }

  Scalar test(std::vector<std::pair<Vector, Vector>> test_data) {
    uint correct = 0;
    for (auto data : test_data) {
      forward(data.first);
      if (onehotdecode(activations.back()) == onehotdecode(data.second)) correct++;
    }
    // std::cout << correct << ' ' << test_data.size() << '\n';
    return (Scalar)correct/(Scalar)test_data.size();
  }

  void SGD(std::vector<std::pair<Vector, Vector>> data, uint batch_size, uint epochs, std::optional<std::vector<std::pair<Vector, Vector>>> test_data = std::optional<std::vector<std::pair<Vector, Vector>>>()) {
    std::random_device rd;
    std::mt19937 g(rd());

    
    for (uint e = 0; e < epochs; e++) {
      // TODO: make it _all_ matrix-based
      // no loop-over-each-example
      for (int i = 0; i < data.size(); i++) {
        Vector input = data[i].first;
        Vector expected = data[i].second;
        forward(input);
        backprop(expected);

        if (i % batch_size == 0 && i != 0) {
          // update stuff
          for(uint i = 0; i < layers - 1; i++) {
            weights[i] -= ((learningRate)/(batch_size)) * delta_weights[i];
            biases[i] -= ((learningRate)/(batch_size)) * delta_biases[i];

            delta_weights[i] *= 0;
            delta_biases[i] *= 0;
          }
        }
      }

      std::shuffle(data.begin(), data.end(), g);
      if (test_data.has_value()) {
        std::shuffle(test_data.value().begin(), test_data.value().end(), g);
        std::cout << "Epoch: " << e << ", accuracy: " << test(test_data.value())*100 << "%\n";
      }
    }
  }

  Vector errorDeriv(Vector expected) {
    return activations.back() - expected;
  }

  void backprop(Vector expected) {
    // find the delta on each neuron
    Vector nabla_cx_d_aL = errorDeriv(expected);
    Vector deltas_last = nabla_cx_d_aL.cwiseProduct(Vectorize<Scalar>(weighted_inps.back(), deriv_fn));
    deltas.back() = deltas_last;
    // if it doesnt work 
    delta_weights.back() += deltas.back() * activations.end()[-2].transpose();
    delta_biases.back() += deltas.back();
    // NOT uint or you WILL overflow
    // there should really be a compiler warning for (uint) >= 0
    for (int i = layers - 3; i >= 0; i--) {
      // std::cout << weights[i + 1].cols() << ' ' << weights[i + 1].rows() << ' ' << deltas[i + 1].rows() << ' ' << deltas[i+1].cols() << '\n';
      // Matrix tmp = (deltas[i+1].transpose() * weights[i + 1]);
      // std::cout << tmp.cols() << ' ' << tmp.rows() << '\n';
      // Matrix sp = Vectorize<Scalar>(weighted_inps[i], deriv_fn);
      // std::cout << sp.cols() << ' ' << sp.rows() << '\n';
      // std::cout << "winps\n";
      // for (auto k : weighted_inps) std::cout << k << "aaa\n";
      // std::cout << weighted_inps[i].rows() << '\n';
      // std::cout << "here\n";
      // Vector out = Vector::Zero(weighted_inps[i].rows());
      // std::cout << "here2\n";
      // for (uint j = 0; j < weighted_inps[i].rows(); j++) {
        // out(j) = deriv_fn(weighted_inps[i](j));
      // }
      // return out;
      deltas[i] = (deltas[i+1].transpose() * weights[i + 1]).transpose().cwiseProduct(Vectorize<Scalar>(weighted_inps[i], deriv_fn));
      // TODO: check this
      Matrix t = (deltas[i] * activations[i].transpose());
      // std::cout << delta_weights[i].rows() << ' ' << delta_weights[i].cols() << ' ' << t.rows() << ' ' << t.cols() << '\n';
      delta_weights[i] += t;
      delta_biases[i] += deltas[i];
    }
  }
};

#endif
