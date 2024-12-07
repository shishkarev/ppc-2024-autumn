// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_pipeline_run) {
  const int rows = 5;
  const int cols = 5;

  // Создание данных для матрицы и вектора правых частей
  std::vector<double> input_matrix = {2.0,  -1.0, 1.0, 3.0, 1.0, 1.0, 3.0, -1.0, 2.0,  -2.0, 3.0, 1.0, 2.0,
                                      -1.0, 4.0,  2.0, 2.0, 3.0, 1.0, 5.0, -1.0, -2.0, 1.0,  3.0, 1.0};
  
  std::vector<double> input_vector_b = {5.0, 10.0, 8.0, 3.0, 7.0};
  std::vector<double> output_vector(5, 0.0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  taskDataSeq->outputs_count.emplace_back(output_vector.size());

  // Создание задачи
  auto gaussianEliminationSequential =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential>(
          taskDataSeq);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;  // Один поток для теста
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussianEliminationSequential);

  // Запуск pipeline и проверка вывода
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка правильности решения системы уравнений
  // Ожидаемый результат можно вычислить заранее или с помощью других средств
  // В данном случае это просто пример, так как решение уравнений нужно проверять отдельно
  const std::vector<double> expected_result = {1.0, -1.0, 2.0, 3.0, 4.0};
  for (size_t i = 0; i < output_vector.size(); ++i) {
    ASSERT_NEAR(output_vector[i], expected_result[i], 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_task_run) {
  const int rows = 5;
  const int cols = 5;

  // Создание данных для матрицы и вектора правых частей
  std::vector<double> input_matrix = {2.0,  -1.0, 1.0, 3.0, 1.0, 1.0, 3.0, -1.0, 2.0,  -2.0, 3.0, 1.0, 2.0,
                                      -1.0, 4.0,  2.0, 2.0, 3.0, 1.0, 5.0, -1.0, -2.0, 1.0,  3.0, 1.0};
  
  std::vector<double> input_vector_b = {5.0, 10.0, 8.0, 3.0, 7.0};
  std::vector<double> output_vector(5, 0.0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  taskDataSeq->inputs_count.emplace_back(sizeof(int));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  taskDataSeq->outputs_count.emplace_back(output_vector.size());

  // Создание задачи
  auto gaussianEliminationSequential =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential>(
          taskDataSeq);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;  // Один поток для теста
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussianEliminationSequential);

  // Запуск задачи и проверка вывода
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка правильности решения системы уравнений
  const std::vector<double> expected_result = {1.0, -1.0, 2.0, 3.0, 4.0};
  for (size_t i = 0; i < output_vector.size(); ++i) {
    ASSERT_NEAR(output_vector[i], expected_result[i], 1e-6);
  }
}
