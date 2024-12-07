#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential::pre_processing() {
  internal_order_test();
  // Подготовка входных данных для последовательной обработки
  matrix = *reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
  vector_b = *reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential::run() {
  internal_order_test();

  // Применяем метод Гаусса для последовательной обработки
  size_t n = matrix.size();

  // Приведение матрицы к верхнетреугольному виду
  for (size_t i = 0; i < n; i++) {
    // Нормализация текущей строки
    double pivot = matrix[i][i];
    for (size_t j = i; j < n; j++) {
      matrix[i][j] /= pivot;
    }
    vector_b[i] /= pivot;

    // Обработка всех строк ниже текущей
    for (size_t j = i + 1; j < n; j++) {
      double factor = matrix[j][i];
      for (size_t k = i; k < n; k++) {
        matrix[j][k] -= factor * matrix[i][k];
      }
      vector_b[j] -= factor * vector_b[i];
    }
  }

  // Обратный ход (решение системы уравнений)
  std::vector<double> solution(n, 0);
  for (int i = n - 1; i >= 0; i--) {
    solution[i] = vector_b[i];
    for (size_t j = i + 1; j < n; j++) {
      solution[i] -= matrix[i][j] * solution[j];
    }
  }

  // Сохраняем решение в outputs
  *reinterpret_cast<std::vector<double>*>(taskData->outputs[0]) = solution;
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential::post_processing() {
  internal_order_test();
  return true;
}