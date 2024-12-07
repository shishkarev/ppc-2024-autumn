// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

using namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi;

TEST(GaussianElimination, test_sequential_simple_system) {
  // Решение системы: 2x + y = 5, x - y = 1
  boost::mpi::environment env;

  std::vector<double> matA = {2.0, 1.0, 1.0, -1.0};  // Матрица коэффициентов (2x2)
  std::vector<double> vecB = {5.0, 1.0};             // Вектор правой части
  int rows = 2;
  int cols = 2;

  std::vector<double> result(2, 0.0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
  taskDataSeq->inputs_count.emplace_back(matA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskDataSeq->inputs_count.emplace_back(vecB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  GaussianEliminationSequential solver(taskDataSeq);

  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  ASSERT_TRUE(solver.run());
  solver.post_processing();

  ASSERT_NEAR(result[0], 2.0, 1e-6);  // x ≈ 2.0
  ASSERT_NEAR(result[1], 1.0, 1e-6);  // y ≈ 1.0
}

TEST(GaussianElimination, test_parallel_simple_system) {
  // Решение системы: 3x + y = 7, x - y = 1
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<double> matA = {3.0, 1.0, 1.0, -1.0};  // Матрица коэффициентов (2x2)
  std::vector<double> vecB = {7.0, 1.0};             // Вектор правой части
  int rows = 2;
  int cols = 2;

  std::vector<double> result(2, 0.0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
    taskDataPar->inputs_count.emplace_back(matA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
    taskDataPar->inputs_count.emplace_back(vecB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  GaussianEliminationParallel solver(taskDataPar);

  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  ASSERT_TRUE(solver.run());
  solver.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(result[0], 2.0, 1e-6);  // x ≈ 2.0
    ASSERT_NEAR(result[1], 1.0, 1e-6);  // y ≈ 1.0
  }
}

TEST(GaussianElimination, test_large_system_parallel) {
  // Генерация случайной системы уравнений
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int n = 100;  // Размер системы (100x100)
  std::vector<double> matA(n * n);
  std::vector<double> vecB(n);
  std::vector<double> expected_result(n);

  if (world.rank() == 0) {
    // Случайная диагонально-доминантная матрица и решение
    for (int i = 0; i < n; ++i) {
      expected_result[i] = i + 1;  // x = [1, 2, ..., n]
      vecB[i] = 0.0;
      for (int j = 0; j < n; ++j) {
        matA[i * n + j] = (i == j) ? 2 * n : 1;  // Диагонально-доминантная матрица
        vecB[i] += matA[i * n + j] * expected_result[j];
      }
    }
  }

  std::vector<double> result(n, 0.0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
    taskDataPar->inputs_count.emplace_back(matA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
    taskDataPar->inputs_count.emplace_back(vecB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  GaussianEliminationParallel solver(taskDataPar);

  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  ASSERT_TRUE(solver.run());
  solver.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result[i], expected_result[i], 1e-6);  // x ≈ expected
    }
  }
}

TEST(GaussianElimination, test_empty_input) {
  // Проверка обработки пустых данных
  boost::mpi::environment env;

  std::vector<double> matA;  // Пустая матрица
  std::vector<double> vecB;  // Пустой вектор
  int rows = 0;
  int cols = 0;

  std::vector<double> result;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
  taskDataSeq->inputs_count.emplace_back(matA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskDataSeq->inputs_count.emplace_back(vecB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  GaussianEliminationSequential solver(taskDataSeq);

  ASSERT_FALSE(solver.validation());  // Должна быть ошибка валидации
}

TEST(GaussianElimination, test_non_square_matrix) {
  // Проверка неквадратной матрицы (ошибка)
  boost::mpi::environment env;

  std::vector<double> matA = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // Матрица 2x3
  std::vector<double> vecB = {7.0, 8.0};
  int rows = 2;
  int cols = 3;

  std::vector<double> result;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
  taskDataSeq->inputs_count.emplace_back(matA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskDataSeq->inputs_count.emplace_back(vecB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  GaussianEliminationSequential solver(taskDataSeq);

  ASSERT_FALSE(solver.validation());  // Должна быть ошибка валидации
}

TEST(GaussianElimination, test_singular_matrix) {
  // Система с сингулярной матрицей (невозможно решить)
  boost::mpi::environment env;

  std::vector<double> matA = {1.0, 2.0, 2.0, 4.0};  // Матрица 2x2 с линейно зависимыми строками
  std::vector<double> vecB = {5.0, 10.0};           // Несогласованный вектор
  int rows = 2;
  int cols = 2;

  std::vector<double> result(2, 0.0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
  taskDataSeq->inputs_count.emplace_back(matA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
  taskDataSeq->inputs_count.emplace_back(vecB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  GaussianEliminationSequential solver(taskDataSeq);

  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  ASSERT_FALSE(solver.run());  // Решение невозможно
}

TEST(GaussianElimination, test_large_sparse_matrix) {
  // Решение системы с разреженной матрицей
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int n = 1000;  // Размер системы (1000x1000)
  std::vector<double> matA(n * n, 0.0);
  std::vector<double> vecB(n, 0.0);
  std::vector<double> expected_result(n, 1.0);  // Решение x = [1, 1, ..., 1]

  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      matA[i * n + i] = 10.0;                    // Диагональные элементы
      if (i < n - 1) matA[i * n + i + 1] = 1.0;  // Соседние элементы
      vecB[i] = 11.0;                            // Обеспечивает решение x[i] = 1
    }
  }

  std::vector<double> result(n, 0.0);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matA.data()));
    taskDataPar->inputs_count.emplace_back(matA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vecB.data()));
    taskDataPar->inputs_count.emplace_back(vecB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  GaussianEliminationParallel solver(taskDataPar);

  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  ASSERT_TRUE(solver.run());
  solver.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(result[i], expected_result[i], 1e-6);  // x ≈ expected
    }
  }
}
