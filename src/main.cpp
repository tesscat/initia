#include <fstream>


#include <Eigen/Eigen>
#include "fns.hpp"
#include "net.hpp"

using Scalar = double;
using Vector = Network<Scalar>::Vector;
using Matrix = Network<Scalar>::Matrix;
using Operator = Network<Scalar>::Operator;

Vector OneHotEncode(uintmax_t Y) {
  Vector vec(10);
  vec.fill(0);

  vec[Y] = 1.0;
  return vec;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str) {
  std::vector<std::string>   result;
  std::string                line;
  std::getline(str,line);

  std::stringstream          lineStream(line);
  std::string                cell;

  while(std::getline(lineStream,cell, ','))
  {
    result.push_back(cell);
  }
  // This checks for a trailing comma with no data after it.
  if (!lineStream && cell.empty())
  {
    // If there was a trailing comma then add an empty element.
    result.push_back("");
  }
  return result;
}

int main() {


  std::ifstream file("mnist_train.csv");

  std::vector<std::pair<Vector, Vector>> data;
  
  // load train data
  while (!file.eof()) {
    std::vector<Scalar> out;
    std::vector<std::string> g = getNextLineAndSplitIntoTokens(file);
    for (auto s : g) {
      if (s == "") continue;
      out.push_back(std::stoi(s));
    }
    // std::cout << out.size() << '\n';
    if (out.size() == 0) break;
    Vector key = OneHotEncode(out[0]);
    // Vector<double> data_(24*24);
    std::vector<Scalar> data_;
    if (out.size() != 28*28 + 1) std::cout << "not right size " << out.size() << '\n';
    for (uintmax_t i = 1; i < out.size(); i++) {
      data_.push_back(out[i]);
    }
    Vector v(data_.size());
    for (uint i = 0; i < data_.size(); i++) {
      v(i) = data_[i];
    }
    data.push_back(std::make_pair(v, key));
  }

  std::ifstream test_file("mnist_train.csv");

  std::vector<std::pair<Vector, Vector>> test_data;
  
  // load test data
  while (!test_file.eof()) {
    std::vector<Scalar> out;
    std::vector<std::string> g = getNextLineAndSplitIntoTokens(test_file);
    for (auto s : g) {
      if (s == "") continue;
      out.push_back(std::stoi(s));
    }
    // std::cout << out.size() << '\n';
    if (out.size() == 0) break;
    Vector key = OneHotEncode(out[0]);
    // Vector<double> data_(24*24);
    std::vector<Scalar> data_;
    if (out.size() != 28*28 + 1) std::cout << "not right size " << out.size() << '\n';
    for (uintmax_t i = 1; i < out.size(); i++) {
      data_.push_back(out[i]);
    }
    Vector v(data_.size());
    for (uint i = 0; i < data_.size(); i++) {
      v(i) = data_[i];
    }
    test_data.push_back(std::make_pair(v, key));
  }
  
  Network<Scalar> net(std::vector<uint>({28*28, 15, 10}), 8, (Operator)Sigmoid<Scalar>, (Operator)SigmoidDeriv<Scalar>);
  std::cout << "beginning SGD\n";
  net.SGD(test_data, 10000, 10, std::optional(test_data));
  std::cout << "finished SDG\n";

  return 0;
}
