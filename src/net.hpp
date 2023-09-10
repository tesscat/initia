#ifndef NET_HPP
#define NET_HPP

#include "fns.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include <CL/sycl.hpp>

template<typename Scalar>
class Network {
public:
  using AtomicScalar = cl::sycl::atomic<Scalar>;
  using Matrix = Eigen::MatrixX<Scalar>;
  using Vector = Eigen::VectorX<Scalar>;
  using AMatrix = Eigen::MatrixX<Scalar>;
  using AVector = Eigen::VectorX<Scalar>;
  using Operator = std::function<Scalar(Scalar)>;
private:
  // std::vector<Vector> activations;
  // std::vector<Vector> weighted_inps;

  std::vector<Matrix> weights;
  std::vector<Vector> biases;
  // std::vector<AMatrix> delta_weights;
  // std::vector<AVector> delta_biases;
  std::vector<uint> topology;
  // std::vector<Vector> deltas;
  Scalar learningRate;
  uint layers;
  Operator fn;
  Operator deriv_fn;
public:
  Network(std::vector<uint> topology_, Scalar rate, Operator fn_, Operator deriv_fn_) : topology(topology_), learningRate(rate), layers(topology.size()), fn(fn_), deriv_fn(deriv_fn_) {
    // activations.push_back(Vector(topology[0]));
    for (int i = 1; i < topology.size(); i++) {
      weights.push_back(Matrix::Zero(topology[i], topology[i - 1]));
      biases.push_back(Vector::Zero(topology[i]));

      // deltas.push_back(Vector::Zero(topology[i]));
      // delta_weights.push_back(Matrix::Zero(topology[i], topology[i - 1]));
      // delta_biases.push_back(Vector::Zero(topology[i]));

      // activations.push_back(Vector::Zero(topology[i]));
      // weighted_inps.push_back(Vector::Zero(topology[i]));
    }

    std::mt19937_64 rng;
    // chosen by dice roll, guaranteed to be random
    rng.seed(4);
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
  std::pair<std::vector<Vector>, std::vector<Vector>> forward(Vector inp) {
    std::vector<Vector> activations;
    std::vector<Vector> weighted_inps;
    Vector activ = inp;
    activations.push_back(activ);
    for (uint i = 0; i < layers - 1; i++) {
      weighted_inps.push_back(weights[i] * activations[i] + biases[i]);
      activations.push_back(Vectorize<Scalar>(weighted_inps.back(), fn));
    }

    return std::make_pair(activations, weighted_inps);
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
      auto p = forward(data.first);
      if (onehotdecode(p.first.back()) == onehotdecode(data.second)) correct++;
    }
    // std::cout << correct << ' ' << test_data.size() << '\n';
    return (Scalar)correct/(Scalar)test_data.size();
  }

  void SGD(std::vector<std::pair<Vector, Vector>> data, uint batch_size, uint epochs, cl::sycl::queue& queue, std::optional<std::vector<std::pair<Vector, Vector>>> test_data = std::optional<std::vector<std::pair<Vector, Vector>>>()) {
    std::random_device rd;
    std::mt19937 g(rd());

    // who needs RAM anyways
    // TODO: see if we can atomic or mutex a cl buffer of matrixes so we can just add the deltas on from each thread
    // std::vector<std::pair<std::vector<Matrix>, std::vector<Vector>>> deltas(batch_size);

    // all the buffers
    std::vector<Matrix> delta_weights_all(batch_size * (layers - 1));
    std::vector<Vector> delta_biases_all(batch_size * (layers - 1));



    // cl::sycl::buffer<Matrix> delta_w_buff(delta_weights.data(), cl::sycl::range<1>(delta_weights.size()));
    // cl::sycl::buffer<Vector> delta_b_buff(delta_biases.data(), cl::sycl::range<1>(delta_biases.size()));
    cl::sycl::buffer<std::pair<Vector, Vector>> data_buff(data.data(), cl::sycl::range<1>(data.size()));

    
    for (uint e = 0; e < epochs; e++) {
      // TODO: make it _all_ matrix-based
      // no loop-over-each-example
      for (int i = 0; i < data.size()/batch_size; i++) {
        {
          cl::sycl::buffer<Matrix, 2> delta_weights_buff(delta_weights_all.data(), cl::sycl::range<2>(batch_size, layers-1));
          cl::sycl::buffer<Vector, 2> delta_biases_buff(delta_biases_all.data(), cl::sycl::range<2>(batch_size, layers-1));
          // new scope cuz SYCL is weird
          // copy things over
          // See, since we do a lot of copying ever iter, we might even lose the benefits of the perf boost from paralleling
          // TODO: can we read from a buffer in host code?

          queue.submit([&](cl::sycl::handler& cgh) {
            // std::cout << "hello from queue\n";
            auto data_acc = data_buff.template get_access<cl::sycl::access::mode::read>(cgh);
            // auto deltaw_acc = delta_w_buff.get_access<cl::sycl::access_mode::atomic>(cgh);
            // auto deltab_acc = delta_b_buff.get_access<cl::sycl::access_mode::atomic>(cgh);
            auto delta_ws_acc = delta_weights_buff.template get_access<cl::sycl::access::mode::read_write>(cgh);
            auto delta_bs_acc = delta_biases_buff.template get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(cl::sycl::range<1>(batch_size), [=](cl::sycl::id<1> pos) {
              // std::cout << "hellp from pfor\n";
              int j = pos.get(0);
              Vector input = data_acc[i * batch_size + j].first;
              Vector expected = data_acc[i * batch_size + j].second;
              backprop(input, expected, delta_ws_acc, delta_bs_acc, j);
            });



            // auto deltas_acc2 = deltas_buff.template get_access<cl::sycl::access::mode::read>(cgh);

            // for (std::pair<std::vector<Matrix>, std::vector<Vector>> p : deltas_acc2) {
            //   for (int i = 0; i < delta_weights.size(); i++) {
            //     delta_weights[i] += p.first[i];
            //     delta_biases[i] += p.second[i];
            //   }
            // }
          });


        }

        // add em up



        // for (int j = 0; j < batch_size; j++) {
        //   // this is the big loop
        //   Vector input = data[i * batch_size + j].first;
        //   Vector expected = data[i * batch_size + j].second;
        //   backprop(input, expected);
        // }
        for (uint i = 0; i < batch_size; i++) {
          for(uint j = 0; j < layers - 1; j++) {
            // TODO: fears over truncation errors with such small thingys
            weights[j] -= ((learningRate)/(batch_size)) * delta_weights_all[(i * (layers - 1)) + j];
            biases[j] -= ((learningRate)/(batch_size)) * delta_biases_all[(i * (layers - 1)) + j];

            delta_weights_all[(i * (layers - 1)) + j] *= 0;
            delta_biases_all[(i * (layers - 1)) + j] *= 0;
          }
        }
      }

      std::shuffle(data.begin(), data.end(), g);
      if (test_data.has_value()) {
        std::shuffle(test_data.value().begin(), test_data.value().end(), g);
        std::cout << "Epoch: " << e << ", accuracy: " << test(test_data.value())*100 << "%\n";
        // std::cout << "weights 0:\n" << weights[0] << '\n';
      }
    }
  }

  Vector errorDeriv(Vector acc, Vector expected) {
    return acc - expected;
  }

  void backprop(Vector inp, Vector expected,
      cl::sycl::accessor<Matrix, 2, cl::sycl::access::mode::read_write> delta_ws_out,
      cl::sycl::accessor<Vector, 2, cl::sycl::access::mode::read_write> delta_bs_out,
      int index) {
    auto p = forward(inp);
    auto activations = p.first;
    auto weighted_inps = p.second;

    // std::vector<Matrix>& deltaw = deltas_out[index].first;
    // std::vector<Vector>& deltab = deltas_out[index].second;

    for (int i = 0; i < weights.size(); i++) {
      delta_ws_out[index][i] = (Matrix::Zero(weights[i].rows(), weights[i].cols()));
      delta_bs_out[index][i] = (Vector::Zero(biases[i].rows()));
    }

    std::vector<Vector> deltas;
    for (int i = 1; i < topology.size(); i++) {
      deltas.push_back(Vector::Zero(topology[i]));
    }

    // find the delta on each neuron
    Vector nabla_cx_d_aL = errorDeriv(activations.back(), expected);
    Vector deltas_last = nabla_cx_d_aL.cwiseProduct(Vectorize<Scalar>(weighted_inps.back(), deriv_fn));
    deltas.back() = deltas_last;

    for (int j = 0; j < deltas_last.cols(); j++) {
        for (int k = 0; k < deltas_last.rows(); k++) {
          if (std::isnan(deltas_last(k, j))) {
            std::cout << "NAN ALERT\n";
            break;
          }
        }
      }
    // if it doesnt work 
    delta_ws_out[index][layers - 2] += deltas.back() * activations.end()[-2].transpose();
    delta_bs_out[index][layers - 2] += deltas.back();
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
      delta_ws_out[index][i] += t;
      delta_bs_out[index][i] += deltas[i];
      // sanity checks plz help
      
    }
  }
};

#endif
