// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationSequential::pre_processing() {
  internal_order_test();

  // Извлечение входных данных
  std::vector<double> input_matrix(taskData->inputs_count[0]);
  double* matrix_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(matrix_ptr, matrix_ptr + input_matrix.size(), input_matrix.begin());

  std::vector<double> input_vector_b(taskData->inputs_count[1]);
  double* vector_b_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(vector_b_ptr, vector_b_ptr + input_vector_b.size(), input_vector_b.begin());

  int rows = *reinterpret_cast<int*>(taskData->inputs[2]);
  int cols = *reinterpret_cast<int*>(taskData->inputs[3]);

  matA = Matrix(input_matrix, rows, cols);
  vecB = input_vector_b;
  result.resize(rows, 0.0);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationSequential::validation() {
  internal_order_test();
  return matA.rows == matA.cols && matA.rows == static_cast<int>(vecB.size());
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationSequential::run() {
  internal_order_test();

  int n = matA.rows;

  // Прямой ход
  for (int k = 0; k < n; ++k) {
    double pivot = matA(k, k);
    if (std::abs(pivot) < 1e-9) return false; // Матрица вырождена
    for (int j = k; j < n; ++j) matA(k, j) /= pivot;
    vecB[k] /= pivot;

    for (int i = k + 1; i < n; ++i) {
      double factor = matA(i, k);
      for (int j = k; j < n; ++j) matA(i, j) -= factor * matA(k, j);
      vecB[i] -= factor * vecB[k];
    }
  }

  // Обратный ход
  for (int i = n - 1; i >= 0; --i) {
    result[i] = vecB[i];
    for (int j = i + 1; j < n; ++j) result[i] -= matA(i, j) * result[j];
  }

  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationSequential::post_processing() {
  internal_order_test();
  double* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), output_ptr);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Инициализация входных данных
    std::vector<double> input_matrix(taskData->inputs_count[0]);
    double* matrix_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(matrix_ptr, matrix_ptr + input_matrix.size(), input_matrix.begin());

    std::vector<double> input_vector_b(taskData->inputs_count[1]);
    double* vector_b_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(vector_b_ptr, vector_b_ptr + input_vector_b.size(), input_vector_b.begin());

    int rows = *reinterpret_cast<int*>(taskData->inputs[2]);
    int cols = *reinterpret_cast<int*>(taskData->inputs[3]);

    matA = Matrix(input_matrix, rows, cols);
    vecB = input_vector_b;
    result.resize(rows, 0.0);
  }

  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return matA.rows == matA.cols && matA.rows == static_cast<int>(vecB.size());
  }
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationParallel::run() {
  internal_order_test();

  // Распределение и сбор данных между процессами
  boost::mpi::broadcast(world, matA.data, 0);
  boost::mpi::broadcast(world, vecB, 0);

  int n = matA.rows;
  int rank = world.rank();
  int size = world.size();

  // Прямой ход
  for (int k = 0; k < n; ++k) {
    if (rank == k % size) {
      double pivot = matA(k / size, k);
      for (int j = k; j < n; ++j) matA(k / size, j) /= pivot;
      vecB[k / size] /= pivot;
    }

    double pivot;
    MPI_Bcast(&pivot, 1, MPI_DOUBLE, k % size, world);

    for (int i = rank; i < n; i += size) {
      if (i > k) {
        double factor = matA(i / size, k);
        for (int j = k; j < n; ++j) matA(i / size, j) -= factor * matA(k / size, j);
        vecB[i / size] -= factor * vecB[k / size];
      }
    }
  }

  // Обратный ход (аналогично)

  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    double* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), output_ptr);
  }
  return true;
}
