// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>

#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int matrix_size = 4;
  std::vector<double> input_matrix;
  std::vector<double> input_vector;
  std::vector<double> result(matrix_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_matrix = {2, 1, -1, -3, -3, -1, 2, 4, 1, 2, 3, 0, 5, 4, 3, 2};
    input_vector = {-8, 13, 10, 5};

    taskDataPar->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input_matrix.data())));
    taskDataPar->inputs_count.emplace_back(input_matrix.size());
    taskDataPar->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input_vector.data())));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&matrix_size)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&matrix_size)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(result.data())));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto parallel_task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationParallel>(
          taskDataPar);

  ASSERT_TRUE(parallel_task->validation());
  parallel_task->pre_processing();

  // Измерение производительности
  auto start_time = std::chrono::high_resolution_clock::now();

  parallel_task->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  if (world.rank() == 0) {
    std::cout << "Pipeline execution time: "
              << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
  }

  parallel_task->post_processing();

  if (world.rank() == 0) {
    std::vector<double> expected_result = {3, -4, -1, 5};
    ASSERT_EQ(result, expected_result);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int matrix_size = 4;
  std::vector<double> input_matrix;
  std::vector<double> input_vector;
  std::vector<double> result(matrix_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_matrix = {2, 1, -1, -3, -3, -1, 2, 4, 1, 2, 3, 0, 5, 4, 3, 2};
    input_vector = {-8, 13, 10, 5};

    taskDataSeq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());
    taskDataSeq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input_vector.data())));
    taskDataSeq->inputs_count.emplace_back(input_vector.size());
    taskDataSeq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&matrix_size)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&matrix_size)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(result.data())));
    taskDataSeq->outputs_count.emplace_back(result.size());
  }

  auto sequential_task =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::GaussianEliminationSequential>(
          taskDataSeq);

  ASSERT_TRUE(sequential_task->validation());
  sequential_task->pre_processing();

  // Измерение производительности
  auto start_time = std::chrono::high_resolution_clock::now();

  sequential_task->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  if (world.rank() == 0) {
    std::cout << "Task execution time: "
              << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
  }

  sequential_task->post_processing();

  if (world.rank() == 0) {
    std::vector<double> expected_result = {3, -4, -1, 5};
    ASSERT_EQ(result, expected_result);
  }
}
