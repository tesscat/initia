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

// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
class CSVRow {
public:
  std::string_view operator[](std::size_t index) const {
    return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
  }
  std::size_t size() const {
    return m_data.size() - 1;
  }
  void readNextRow(std::istream& str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while((pos = m_line.find(',', pos)) != std::string::npos) {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos   = m_line.size();
    m_data.emplace_back(pos);
  }
private:
  std::string         m_line;
  std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
  data.readNextRow(str);
  return str;
}

Vector<double> OneHotEncode(uintmax_t Y) {
  Vector<double> vec(10);
  vec.fill(0);

  vec[Y] = 1.0;
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

  Network<double> net({28*28, 15, 10});

  std::ifstream file("mnist.csv");

  std::vector<std::pair<Vector<double>, Vector<double>>> data;
  
  // CSVRow row;
  while (!file.eof()) {
    std::vector<double> out;
    std::vector<std::string> g = getNextLineAndSplitIntoTokens(file);
    for (auto s : g) {
      if (s == "") continue;
      out.push_back(std::stoi(s));
    }
    // std::cout << out.size() << '\n';
    if (out.size() == 0) break;
    Vector<double> key = OneHotEncode(out[0]);
    // Vector<double> data_(24*24);
    std::vector<double> data_;
    if (out.size() != 28*28 + 1) std::cout << "not right size " << out.size() << '\n';
    for (uintmax_t i = 1; i < out.size(); i++) {
      data_.push_back(out[i]);
    }
    data.push_back(std::make_pair(Vector<double>(data_), key));
  }
  
  std::cout << "beginning SGD\n";
  net.StochGradDesc(data, 5, 100, 0);
  std::cout << "finished SDG\n";

  
  // vec_from<int, 4>(l1.fwd()).print(std::cout);
  
  // net.test();

  // net.Run(vec_from<double, 8>({1, 1, 1, 1, 1, 1, 1, 1})).print(std::cout);
  return 0;
}
