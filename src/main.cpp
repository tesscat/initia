#include "fns.hpp"
#include "mtx.hpp"
#include "layer.hpp"
#include "net.hpp"
#include <iostream>
#include <random>


int main() {

  std::mt19937_64 gen;
  // chosen by dice roll, guaranteed to be random
  gen.seed(5);

  std::uniform_real_distribution<double> dis(0.0, 1.0);

  std::function<double()> generator = [&](){return dis(gen);};

  InputLayer<double, 2> inl;
  InternalLayer<double, (activfn_t<double>)fns::ATan, (activfn_t<double>)fns::Deriv_ATan, 2, 2> l1(inl);
  InternalLayer<double, (activfn_t<double>)fns::ATan, (activfn_t<double>)fns::Deriv_ATan, 1, 2> l2(l1);
  
  l1.weights.fill(0);
  l1.weights[0][0] = 1;
  l1.weights[1][1] = 1;

  // l1.GenBiases(generator);
  l1.GenWeights(generator);
  l1.GenBiases(generator);
  l2.GenWeights(generator);
  l2.GenBiases(generator);

  l1.weights.print(std::cout);

  // inl.set(vec_from<double, 8>({1, 1, 1, 1, 1, 1, 1, 1}));
  std::vector<Layer<double>*> layers ({&inl, &l1, &l2});

  // static constexpr uintmax_t nodes[] = {8, 4};
  // static constexpr activfn_t<int> fns[] = {fns::ReLu};

  Network<double, 3, 2, 1> net(inl, layers);

  // try and get it to uhh
  // classify a diagonal line ig
  std::function<double(double, double)> truth = [&](double x, double y){
    return x + y;
  };
  
  
  double running_acc = 0.5;
  for (uintmax_t i = 1; true; i++) {
    // train for 40, test for 10
    for (uintmax_t train = 0; train < 40; train++) {
      double v1 = generator();
      double v2 = generator();
      net.AccumulateBackprop(Vector<double, 2>({v1, v2}), Vector<double, 1>({truth(v1, v2)}));
    }
    net.ApplyNudges(1, 40);
    // test for 10
    uintmax_t correct = 0;
    for (uintmax_t i = 0; i < 10; i++) {
      double v1 = generator();
      double v2 = generator();
      // 5% tolerance ig
      if (std::abs((net.Run(Vector<double, 2>{v1, v2})[0] - truth(v1, v2))/truth(v1, v2)) < 0.05) correct++;
      // if (std::round(net.Run(Vector<double, 2>{v1, v2})[0]) == truth(v1, v2)) correct++;
    }
    running_acc = (running_acc * (i-1) + ((double)correct/10))/i;
    std::cout << i*40 << " tests, currently at " << correct << "/10 accuracy with " << running_acc << " running\n";
    l1.weights.print(std::cout);
    l1.biases.print(std::cout);
    l2.weights.print(std::cout);
    l2.biases.print(std::cout);
    std::cin.get();
  }

  // vec_from<int, 4>(l1.fwd()).print(std::cout);
  
  // net.test();

  // net.Run(vec_from<double, 8>({1, 1, 1, 1, 1, 1, 1, 1})).print(std::cout);
  return 0;
}
