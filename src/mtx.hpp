#ifndef MTX_HPP
#define MTX_HPP

#include <functional>
#include <initializer_list>
#include <iostream>
#include <vector>
#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>

template <typename T>
using validfn_t = std::function<T(T)>;
template<typename T>
class Vector;

template <typename T>
class Matrix {
private:

public:
  uintmax_t height;
  uintmax_t width;
  std::vector<std::vector<T>> data;

  Matrix() = delete;

  Matrix(uintmax_t h, uintmax_t w) : width(w), height(h) {
    for (uintmax_t i = 0; i<h; i++) {
      data.push_back(std::vector<T>(w));
    }
    fill(0);
  }

  Matrix(const Matrix<T>* m) : height(m->height), width(m->width) {
    data = (m->data);
  }

  Matrix<T> Transpose() {
    Matrix<T> out(width, height);
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        out.data[x][y] = data[y][x];
      }
    }

    return out;
  }
  // Matrix(const Matrix& other) { std::copy(other.data.begin, other.data.end, data); };
  // Matrix(const Matrix&& other) : data (std::move(other.data)) {};

  void fill(T i) {
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        data[y][x] = i;
      }
    }
  }

  void fill(std::function<T()> fn) {
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        data[y][x] = fn();
      }
    }
  }

  
  void foreach(validfn_t<T> f) {
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        data[y][x] = f(data[y][x]);
      }
    }
  }

  Matrix<T>* operator=(const Matrix<T> other) {
    data = (other.data);
    width = other.width;
    height = other.height;

    return this;
  }

  // std::vector<T> operator[](const uintmax_t y) {return data[y];}
  // virtual const std::vector<T> operator[](const uintmax_t y) const {return data[y];}

  void operator +=(const Matrix<T>& other) {
    if (width != other.width || height != other.height)
      throw std::domain_error("width/height not matched on matrix +");
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        data[y][x] += other.data[y][x];
      }
    }
  }
  const Matrix<T> operator +(const Matrix<T>& other) const {
    Matrix<T> m = this;
    m += other;
    return m;
  }

  void operator *=(const T val) {
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        // hack to stop nans from spreading
        // TODO: remove
        data[y][x] = (val == 0? 0 : val*data[y][x]);
      }
    }
  }

  Matrix<T> operator *(const T val) {
    Matrix<T> q = this;
    q *= val;
    return q;
  }
  
  // we take our height and their width
  Matrix<T> operator *(const Matrix<T>& other) {
    if (width != other.height) {
      std::cout << "my w " << width << " vs " << other.height << '\n';
      throw std::domain_error("dims mismatch on matrix *");
    }
    Matrix<T> m(height, other.width);
    for (uintmax_t i = 0; i < height; i++) {
      for (uintmax_t j = 0; j < other.width; j++) {
        T sum = 0;
        for (uintmax_t k = 0; k < width; k++) {
          sum += data[i][k] * other.data[k][j];
        }
        m.data[i][j] = sum;
      }
    }
    return m;
  }

  Matrix<T> operator *(const std::vector<T>& other) {
    if (width != other.size())
      throw std::domain_error("dims mismatch on m/vec *");
    Matrix<T> m;
    for (uintmax_t j = 0; j < height; j++) {
      T sum = 0;
      for (uintmax_t k = 0; k < width; k++) {
        sum += other[k] * data[j][k];
      }
      m.data[j][0] = sum;
    }
    return m;
  } 

  void print(std::ostream& os) { 
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        os << data[y][x] << ' ';
      }
      os << '\n';
    }
  }

  Vector<T> AsVec() {
    if (width != 1)
      throw new std::domain_error("width != 1");
    return Vector<T>(this);
  }
};


template<typename T>
class Vector : public Matrix<T> {
public:
  Vector() = delete;
  Vector(const Vector<T>&) = default;
  Vector(Vector<T>&&) = default;
  Vector(Matrix<T> m) : Matrix<T>(m) {};
  Vector(uintmax_t h) : Matrix<T>(h, 1) {};
  Vector(std::vector<T> v) : Matrix<T>(v.size(), 1) {
    for (uintmax_t i = 0; i < v.size(); i++) {
      this->data[i][0] = v[i];
    }
  }

  Vector(std::initializer_list<T> q) : Matrix<T>(q.size(), 1) {
    uintmax_t i = 0;
    for (auto k : q) {
      this->data[i][0] = k;
      i++;
    }
  }
  Vector<T> Hadamond(Vector<T> other) {
    Vector<T> out (other.height);
    for (uintmax_t i = 0; i < other.height; i++) {
      out.data[i][0] = this->data[i][0] * other.data[i][0];
    }

    return out;
  }

  Vector<T>* operator=(const Vector<T> other) {
    this->data = other.data;
    this->width = other.width;
    this->height = other.height;

    return this;
  }
};

template<typename T>
void print_vec(std::vector<T> vec, std::ostream& os) {
  for (T k : vec) {
    os << k << '\n';
  }
}

#include<random>
namespace mtx::tests {
void hadamond() {
  std::mt19937_64 rng = std::mt19937_64();
  std::vector<double> v;
  for (int i = 0; i < 10; i++)
    v.push_back((double)(rng()/(rng.max()/10)));
  Vector<double> v1(v);
  for (int i = 0; i < 10; i++)
    v[i] = ((double)(rng()/(rng.max()/10)));
  Vector<double> v2(v);
  v1.print(std::cout);
  std::cout << "hadamond\n";
  v2.print(std::cout);
  std::cout << "=\n";
  (v1.Hadamond(v2)).print(std::cout);
}
void multi() {
  std::mt19937_64 rng = std::mt19937_64();
  std::function<double(double)> r = [&](double b) {
    return (double)(rng()/(rng.max()/10));
  };
  Matrix<double> m1 (2, 6);
  Matrix<double> m2 (6, 4);
  m1.foreach(r);
  m2.foreach(r);
  m1.print(std::cout);
  std::cout << "*\n";
  m2.print(std::cout);
  std::cout << "=\n";
  (m1*m2).print(std::cout);
}
void plus() {
  std::mt19937_64 rng = std::mt19937_64();
  std::function<double(double)> r = [&](double b) {
    return (double)(rng()/(rng.max()/10));
  };
  Matrix<double> m1 (2, 6);
  Matrix<double> m2 (2, 6);
  m1.foreach(r);
  m2.foreach(r);
  m1.print(std::cout);
  std::cout << "+\n";
  m2.print(std::cout);
  std::cout << "=\n";
  m1 += m2;
  (m1).print(std::cout);
}
void trans() {
  std::mt19937_64 rng = std::mt19937_64();
  std::function<double(double)> r = [&](double b) {
    return (double)(rng()/(rng.max()/10));
  };
  Matrix<double> m1 (2, 6);
  m1.foreach(r);
  m1.print(std::cout);
  std::cout << "T=\n";
  (m1.Transpose()).print(std::cout);
}
};

// template<typename T, const uintmax_t H>
// class Vector : Matrix<T, 1, H> {
// public:
//   T* operator[](const uintmax_t x) override {return data[1][x];}
//   const T* operator[](const uintmax_t x) const override {return data[1][x];}
// };
//
#endif
