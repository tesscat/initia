#ifndef NET_HPP
#define NET_HPP

#include "fns.hpp"
#include "mtx.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

using usize = uintmax_t;

template<typename T>
class Network {
  std::vector<Matrix<T>> weights;
  std::vector<Vector<T>> biases;

  std::mt19937_64 rng;

  // TODO: std::vector<std::function<T(T)>> squashes;
public:

  Network(std::vector<uintmax_t> layersizes) {

    // std::mt19937_64 gen;
    rng = std::mt19937_64();
    // chosen by dice roll, guaranteed to be random
    rng.seed(5);
    std::uniform_real_distribution<T> dis(-1.0, 1.0);
    std::function<T(T)> generator = [&](T _){return dis(rng);};

    for (uintmax_t i = 1; i < layersizes.size(); i++) {
      Matrix<T> mtx(layersizes[i], layersizes[i - 1]);
      mtx.foreach(generator);
      weights.push_back(mtx);
      Vector<T> vec(layersizes[i]);
      vec.foreach((validfn_t<T>)generator);
      biases.push_back(vec);
    }
  }

  // Vector<T> Run(Vector<T> inp) {
  //   Vector<T> v = inp;
  //   for (usize i = 0; i < weights.size(); i++) {
  //     v = weights[i] * v;
  //     v += biases[i];
  //     v.foreach(squashes[i]);
  //   }
  //
  //   return v;
  // }

  T Test(std::vector<std::pair<Vector<T>, Vector<T>>>& training_data, uintmax_t count) {
    uintmax_t correct = 0;
    for (uintmax_t i = 0; i < count; i++) {
      Vector<T> inp = training_data[i].first;
      Vector<T> activation = inp;

      for (usize i = 0; i < weights.size(); i++) {
        // weights[i].print(std::cout);
        Vector<T> z = weights[i] * activation + biases[i];
        activation = fns::Sigmoid(z);
      }

      activation.print(std::cout);

      // error
      // assumes MNIST/classification, TODO: generalise
      uintmax_t max_acc_idx = 0;
      T max = -4000;
      for (uintmax_t j = 0; j < activation.height; j++) {
        if (activation.data[j][0] > max) {
          max = activation.data[j][0];
          max_acc_idx = j;
        }
      }
      std::cout << "MAI" << max_acc_idx << '\n';

      uintmax_t max_exp_idx = 0;
      max = -4000;
      for (uintmax_t j = 0; j < activation.height; j++) {
        if (training_data[i].second.data[j][0] > max) {
          max = training_data[i].second.data[j][0];
          max_acc_idx = j;
        }
      }

      if (max_acc_idx == max_exp_idx) {
        correct++;
      }
    }

    return (T)correct/(T)count;
  }

  void StochGradDesc(std::vector<std::pair<Vector<T>, Vector<T>>>& training_data, usize epochs, usize mini_batch_size, T learning_rate) {
    for (usize i = 0; i < epochs; i++) {
      Matrix<T> wb = weights[0] * (-1.0);
      // TODO: shuffle data
      std::shuffle(training_data.begin(), training_data.end(), rng);
      for (usize batch = 0; batch < training_data.size()/mini_batch_size; batch++) {
        // std::vector<std::pair<Vector<T>, Vector<T>>> minibatches = 
        UpdateMiniBatch(training_data, mini_batch_size, batch, learning_rate);
      }
      // std::cout << "dumping params\n";
      // for(auto w : weights) {
        // std::cout << w.height << ' ' << w.width << '\n';
        // w.print(std::cout);
        // std::cout << '\n';
      // }
      // std::cout << "biases\n";
      // for(auto b : biases) {
        // std::cout << b.height << ' ' << b.width << '\n';
        // b.print(std::cout);
        // std::cout << '\n';
      // }
      Matrix<T> wa = weights[0] + wb;
      // wa.print(std::cout);
      std::cout << "Epoch run, accuracy is " << Test(training_data, 50) << std::endl;
    }
  }

  T Cost(T acc, T exp) {
    return std::pow(0.5*(acc - exp), 2);
  }
  Vector<T> Cost(Vector<T> acc, Vector<T> exp) {
    Vector<T> out (acc.height);
    for (uintmax_t i = 0; i < acc.height; i++) {
      out.data[i][0] = Cost(acc.data[i][0], exp.data[i][0]);
    }

    return out;
  }

  T CostDeriv(T acc, T exp) {
    return acc - exp;
  }
  Vector<T> CostDeriv(Vector<T> acc, Vector<T> exp) {
    Vector<T> out (acc.height);
    for (uintmax_t i = 0; i < acc.height; i++) {
      out.data[i][0] = CostDeriv(acc.data[i][0], exp.data[i][0]);
    }

    return out;
  }

  void UpdateMiniBatch(std::vector<std::pair<Vector<T>, Vector<T>>>& training_data, usize mini_batch_size, usize batch, T learning_rate) {
    std::vector<Vector<T>> d_cost_wrt_bias;
    std::vector<Matrix<T>> d_cost_wrt_weight;
    for (usize i = 0; i < weights.size(); i++) {
      d_cost_wrt_bias.push_back(Vector<T>(biases[i].height));
      d_cost_wrt_weight.push_back(Matrix<T>(weights[i].height, weights[i].width));
    }
    for (usize j = batch * mini_batch_size; j<(batch+1)*mini_batch_size; j++) {
      // std::cout << "bpropping\n";
      auto delta_params = Backprop(training_data[j].first, training_data[j].second);
      for (usize i = 0; i < d_cost_wrt_bias.size(); i++) {
        d_cost_wrt_bias[i] += delta_params.second[i];
        d_cost_wrt_weight[i] += delta_params.first[i];
      }
    }
    for (usize i = 0; i < d_cost_wrt_bias.size(); i++) {
      T rate = ((-1.0) * ((T)learning_rate/(T)mini_batch_size));
      // T sum = 0;
      // std::function<T(T)> fn = [&](T i) {
        // sum += i;
        // return i;
      // };
      // std::cout << "dcostwrtweight\n";
      // d_cost_wrt_weight[i].print(std::cout);
      // d_cost_wrt_weight[i].foreach(fn);
      // std::cout << "promise " << sum << '\n';
      // sum = 0;
      // d_cost_wrt_bias[i].foreach(fn);
      // std::cout << "bias " << sum << '\n';
      // std::cout << rate << '\n';
      // std::cout << "rate: " << rate << '\n';
      weights[i] += d_cost_wrt_weight[i] * rate;
      biases[i] += d_cost_wrt_bias[i] * rate;
    }
  }
  
  // return a pair with the derivatives of cost WRT all the weights, and all the biases
  std::pair<std::vector<Matrix<T>>, std::vector<Vector<T>>> Backprop(Vector<T> inp, Vector<T> expected) {
    // std::cout << "inp shape " << inp.height << ' ' << inp.width << '\n';
    // TODO: make it do all the inps/exps in a batch simultaneously
    std::function<T(T)> fn = [](T _) {return 10.0;};
    std::vector<Vector<T>> nabla_bias;
    std::vector<Matrix<T>> nabla_weight;
    for (usize i = 0; i < weights.size(); i++) {
      nabla_bias.push_back(Vector<T>(biases[i].height));
      // nabla_bias[i].foreach(fn);
      nabla_weight.push_back(Matrix<T>(weights[i].height, weights[i].width));
      // nabla_weight[i].foreach(fn);
    }
    // return std::make_pair(nabla_weight, nabla_bias);

    // feed-forward
    std::vector<Vector<T>> activations {};
    Vector<T> activation = inp;
    activations.push_back(activation);
    // activations.end()[-1].print(std::cout);

    std::vector<Vector<T>> weighted_inps {};
    for (usize i = 0; i < weights.size(); i++) {
      // weights[i].print(std::cout);
      Vector<T> z = weights[i] * activation + biases[i];
      // std::cout << "z shape " << z.height << ' ' << z.width << '\n';
      // z.print(std::cout);
      weighted_inps.push_back(z);
      activation = fns::Sigmoid(z);
      // activation.print(std::cout);
      // std::cin.get();
      activations.push_back(activation);
    }

    // backwards pass
    Vector<T> delta = CostDeriv(activations.end()[-1], expected);
    delta = delta.Hadamond(fns::SigmoidPrime(weighted_inps.end()[-1]));
    nabla_bias.end()[-1] = delta;
    nabla_weight.end()[-1] = delta * (activations.end()[-2].Transpose());

    for (uintmax_t l = 2; l < weights.size() + 1; l++) {
      Vector<T> z = weighted_inps.end()[-l];
      Vector<T> sp = fns::SigmoidPrime(z);
      // delta is of next layer atm
      delta = ((weights.end()[-l+1].Transpose())*delta).AsVec().Hadamond(sp);
      // delta.print(std::cout);
      nabla_bias.end()[-l] = delta;
      nabla_weight.end()[-l] = delta * (activations.end()[-l-1].Transpose());
    }
    // std::cout << "nabla weight 0\n";
    // for(int i = 0; i<nabla_weight.size(); i++) {
      // nabla_weight[i].print(std::cout);
      // nabla_bias[i].print(std::cout);
    // }
    return std::make_pair(nabla_weight, nabla_bias);
  }

  // Vector<T> Cost(Ve)
};

#endif // !NET_HPP
