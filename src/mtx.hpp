#ifndef MTX_HPP
#define MTX_HPP

#include <functional>
#include <initializer_list>
#include <vector>
#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>

template <typename T, const uintmax_t W, const uintmax_t H>
class Matrix {
private:
  using validfn_t = T(T);

public:
  uintmax_t width = W;
  uintmax_t height = H;
  uintmax_t size = W*H;
  T data[W][H]; // by rows
  
  T** get_data() {return data;};

  Matrix() {
    fill(0);
  }
  Matrix(T i) {fill(i);}
  // Matrix(const Matrix& other) { std::copy(other.data.begin, other.data.end, data); };
  // Matrix(const Matrix&& other) : data (std::move(other.data)) {};

  void fill(T i) {
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        data[x][y] = i;
      }
    }
  }

  void fill(std::function<T()> fn) {
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        data[x][y] = fn();
      }
    }
  }

  
  void foreach(validfn_t f) {
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        data[x][y] = f(data[x][y]);
      }
    }
  }

  T* operator[](const uintmax_t x) {return data[x];}
  virtual const T* operator[](const uintmax_t x) const {return data[x];}

  void operator +=(const Matrix<T, W, H>& other) {
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        data[x][y] += other.data[x][y];
      }
    }
  }
  const Matrix<T, W, H> operator +(const Matrix<T, W, H>& other) const {
    Matrix<T, W, H> m = this;
    m += other;
    return m;
  }

  void operator *=(const T val) {
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        data[x][y] *= val;
      }
    }
  }

  Matrix<T, W, H> operator *(const T val) {
    Matrix<T, W, H> q = std::copy(this);
    q *= val;
    return q;
  }
  
  // we take our height and their width
  template<const uintmax_t their_W>
  Matrix<T, their_W, H> operator *(const Matrix<T, their_W, W>& other) {
    Matrix<T, their_W, H> m;
    for (uintmax_t i = 0; i < their_W; i++) {
      for (uintmax_t j = 0; j < H; j++) {
        T sum = 0;
        for (uintmax_t k = 0; k < W; k++) {
          sum += other[i][k] * data[k][j];
        }
        m.data[i][j] = sum;
      }
    }
    return m;
  }

  Matrix<T, 1, H> operator *(const std::vector<T>& other) {
    Matrix<T, 1, H> m;
    for (uintmax_t j = 0; j < H; j++) {
      T sum = 0;
      for (uintmax_t k = 0; k < W; k++) {
        sum += other[k] * data[k][j];
      }
      m.data[0][j] = sum;
    }
    return m;
  } 

  void print(std::ostream& os) { 
    for (uintmax_t y = 0; y < H; y++) {
      for (uintmax_t x = 0; x < W; x++) {
        os << data[x][y] << ' ';
      }
      os << '\n';
    }
  }
};


template<typename T, const uintmax_t H>
class Vector : public Matrix<T, 1, H> {
public:
  Vector() : Matrix<T, 1, H>() {};
  Vector(Matrix<T, 1, H> m) : Matrix<T, 1, H>(m) {};
  Vector(std::initializer_list<T> q) {
    uintmax_t i = 0;
    for (auto k : q) {
      this->data[0][i] = k;
      i++;
    }
  }
  std::vector<T> flatten() {
    std::vector<T> v;
    v.insert(v.begin(), std::begin(this->data[0]), std::end(this->data[0]));
    return v;
  }
  T& operator [](uintmax_t idx) {
    return this->data[0][idx];
  }
};

// TODO: enforce that q.size == S
template<typename T, const uintmax_t S>
constexpr Vector<T, S> vec_from(std::initializer_list<T> q) {
  Vector<T, S> v;
  uintmax_t i = 0;
  for (auto k : q) {
    v.data[0][i] = k;
    i++;
  }
  return v;
}
template<typename T, const uintmax_t S>
constexpr Vector<T, S> vec_from(std::vector<T> q) {
  Vector<T, S> v;
  uintmax_t i = 0;
  for (auto k : q) {
    v.data[0][i] = k;
    i++;
  }
  return v;
}

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
