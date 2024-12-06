// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_int) {
  // Create input data
  const int rows = 3;
  const int cols = 3;
  
  // Define input matrix and vector b
  std::vector<int> input_matrix = {
      2, 1, -1,
      -3, -1, 2,
      -2, 1, 2
  };
  std::vector<int> input_vector_b = {8, -11, -3};
  std::vector<int> output_data(rows, 0);  // Output vector for results

  // Define expected result (solving the system of equations)
  std::vector<int> expected_result = {2, 3, -1};

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data->inputs_count.emplace_back(input_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  task_data->inputs_count.emplace_back(input_vector_b.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.emplace_back(sizeof(rows));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.emplace_back(sizeof(cols));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create and run Task
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Validate the output
  for (int i = 0; i < rows; ++i) {
    EXPECT_EQ(output_data[i], expected_result[i]);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_float) {
  // Create input data
  const int rows = 3;
  const int cols = 3;

  // Define input matrix and vector b
  std::vector<float> input_matrix = {
      2.f, 1.f, -1.f,
      -3.f, -1.f, 2.f,
      -2.f, 1.f, 2.f
  };
  std::vector<float> input_vector_b = {8.f, -11.f, -3.f};
  std::vector<float> output_data(rows, 0.f);  // Output vector for results

  // Define expected result (solving the system of equations)
  std::vector<float> expected_result = {2.f, 3.f, -1.f};

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data->inputs_count.emplace_back(input_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  task_data->inputs_count.emplace_back(input_vector_b.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.emplace_back(sizeof(rows));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.emplace_back(sizeof(cols));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create and run Task
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Validate the output
  for (int i = 0; i < rows; ++i) {
    EXPECT_NEAR(output_data[i], expected_result[i], 1e-6f);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_double) {
  // Create input data
  const int rows = 3;
  const int cols = 3;

  // Define input matrix and vector b
  std::vector<double> input_matrix = {
      2.0, 1.0, -1.0,
      -3.0, -1.0, 2.0,
      -2.0, 1.0, 2.0
  };
  std::vector<double> input_vector_b = {8.0, -11.0, -3.0};
  std::vector<double> output_data(rows, 0.0);  // Output vector for results

  // Define expected result (solving the system of equations)
  std::vector<double> expected_result = {2.0, 3.0, -1.0};

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data->inputs_count.emplace_back(input_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  task_data->inputs_count.emplace_back(input_vector_b.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.emplace_back(sizeof(rows));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.emplace_back(sizeof(cols));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create and run Task
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Validate the output
  for (int i = 0; i < rows; ++i) {
    EXPECT_NEAR(output_data[i], expected_result[i], 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_int64_t) {
  // Create input data
  const int rows = 3;
  const int cols = 3;

  // Define input matrix and vector b
  std::vector<int64_t> input_matrix = {
      2, 1, -1,
      -3, -1, 2,
      -2, 1, 2
  };
  std::vector<int64_t> input_vector_b = {8, -11, -3};
  std::vector<int64_t> output_data(rows, 0);  // Output vector for results

  // Define expected result (solving the system of equations)
  std::vector<int64_t> expected_result = {2, 3, -1};

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data->inputs_count.emplace_back(input_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  task_data->inputs_count.emplace_back(input_vector_b.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.emplace_back(sizeof(rows));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.emplace_back(sizeof(cols));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create and run Task
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Validate the output
  for (int i = 0; i < rows; ++i) {
    EXPECT_EQ(output_data[i], expected_result[i]);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_uint8_t) {
  // Create input data
  const int rows = 3;
  const int cols = 3;

  // Define input matrix and vector b
  std::vector<uint8_t> input_matrix = {
      2, 1, 1,
      3, 1, 2,
      2, 2, 2
  };
  std::vector<uint8_t> input_vector_b = {6, 12, 10};
  std::vector<uint8_t> output_data(rows, 0);  // Output vector for results

  // Define expected result (solving the system of equations)
  std::vector<uint8_t> expected_result = {2, 3, 1};

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data->inputs_count.emplace_back(input_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_b.data()));
  task_data->inputs_count.emplace_back(input_vector_b.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
  task_data->inputs_count.emplace_back(sizeof(rows));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
  task_data->inputs_count.emplace_back(sizeof(cols));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  // Create and run Task
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::GaussianEliminationSequential task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  // Validate the output
  for (int i = 0; i < rows; ++i) {
    EXPECT_EQ(output_data[i], expected_result[i]);
  }
}

