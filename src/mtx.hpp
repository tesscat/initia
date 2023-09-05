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
    data = m->data;
  }

  Matrix<T> Transpose() {
    Matrix out(width, height);
    for (uintmax_t y = 0; y < height; y++) {
      for (uintmax_t x = 0; x < width; x++) {
        out[x][y] = data[y][x];
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
    data = other.data;
    width = other.width;
    height = other.height;

    return this;
  }

  std::vector<T> operator[](const uintmax_t y) {return data[y];}
  virtual const std::vector<T> operator[](const uintmax_t y) const {return data[y];}

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
        data[y][x] *= val;
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
    if (height != other.width) {
      std::cout << "my h " << height << " vs " << other.width << '\n';
      throw std::domain_error("dims mismatch on matrix *");
    }
    Matrix<T> m(other.width, height);
    for (uintmax_t i = 0; i < other.width; i++) {
      for (uintmax_t j = 0; j < height; j++) {
        T sum = 0;
        for (uintmax_t k = 0; k < width; k++) {
          sum += other[k][i] * data[j][k];
        }
        m.data[j][i] = sum;
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

  T& operator [](uintmax_t idx) {
    return this->data[idx][0];
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
      out[i] = this[i][0] * other[i];
    }

    return out;
  }

  Vector<T>* operator=(const Vector<T> other) {
    this->data = other.data;
    this->width = other.width;
    this->height = other.height;

    return this;
  }
  // std::vector<T> flatten() {
  //   std::vector<T> v;
  //   v.insert(v.begin(), std::begin(this->data[0]), std::end(this->data[0]));
  //   return v;
  // }
};

template<typename T>
void print_vec(std::vector<T> vec, std::ostream& os) {
  for (T k : vec) {
    os << k << '\n';
  }
}

// template<typename T, const uintmax_t H>
// class Vector : Matrix<T, 1, H> {
// public:
//   T* operator[](const uintmax_t x) override {return data[1][x];}
//   const T* operator[](const uintmax_t x) const override {return data[1][x];}
// };
//
#endif
