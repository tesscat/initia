#include "fns.hpp"
#include "mtx.hpp"
#include "net.hpp"
#include <charconv>
#include <iostream>
#include <random>

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <string>


using T = float;

Vector<T> OneHotEncode(uintmax_t Y) {
  Vector<T> vec(10);
  vec.fill(0);

  vec.data[Y][0] = 1.0;
  return vec;
}

uintmax_t SVTo(std::string sv) {
  uintmax_t i;
  auto result = std::from_chars(sv.data(), sv.data() + sv.size(), i);
  if (result.ec == std::errc::invalid_argument) {
    std::cout << "Could not convert.";
  }

  return i;
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
  // mtx::tests::plus();
  // exit(0);
  // fns::tests::sigm();
  // exit(0);

  Network<T> net({28*28, 15, 10});

  std::ifstream file("mnist.csv");

  std::vector<std::pair<Vector<T>, Vector<T>>> data;
  
  // CSVRow row;
  while (!file.eof()) {
    std::vector<T> out;
    std::vector<std::string> g = getNextLineAndSplitIntoTokens(file);
    for (auto s : g) {
      if (s == "") continue;
      out.push_back(std::stoi(s));
    }
    // std::cout << out.size() << '\n';
    if (out.size() == 0) break;
    Vector<T> key = OneHotEncode(out[0]);
    // Vector<T> data_(24*24);
    std::vector<T> data_;
    if (out.size() != 28*28 + 1) std::cout << "not right size " << out.size() << '\n';
    for (uintmax_t i = 1; i < out.size(); i++) {
      data_.push_back(out[i]);
    }
    data.push_back(std::make_pair(Vector<T>(data_), key));
  }
  
  std::cout << "beginning SGD\n";
  net.StochGradDesc(data, 5, 1000, 300);
  std::cout << "finished SDG\n";

  
  // vec_from<int, 4>(l1.fwd()).print(std::cout);
  
  // net.test();

  // net.Run(vec_from<T, 8>({1, 1, 1, 1, 1, 1, 1, 1})).print(std::cout);
  return 0;
}
