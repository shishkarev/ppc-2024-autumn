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

  // Проверяем наличие taskData и его содержимое
  if (!taskData || taskData->inputs.size() < 2 || taskData->outputs.empty()) {
    return false;  // Недостаточно входных или выходных данных
  }

  // Проверяем корректность матрицы
  auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
  auto* input_vector = reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  auto* output_vector = reinterpret_cast<std::vector<double>*>(taskData->outputs[0]);

  if (input_matrix == nullptr || input_matrix->empty()) {
    return false;  // Матрица отсутствует или пуста
  }

  size_t matrix_size = input_matrix->size();
  for (const auto& row : *input_matrix) {
    if (row.size() != matrix_size) {
      return false;  // Матрица должна быть квадратной
    }
  }

  if (input_vector == nullptr || input_vector->size() != matrix_size) {
    return false;  // Размер вектора должен совпадать с размером матрицы
  }

  if (output_vector == nullptr || output_vector->size() != matrix_size) {
    return false;  // Размер выходного вектора должен совпадать с размером матрицы
  }

  return true;  // Валидация пройдена
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