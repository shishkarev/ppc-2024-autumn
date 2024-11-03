// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

TEST(shishkarev_a_sum_of_vector_elements_seq, test_int) {
  // Create data
  std::vector<int32_t> in(1, 10);
  const int expected_sum = 10;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_float) {
  // Create data
  std::vector<float> in(1, 1.f);
  std::vector<float> out(1, 0.f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<float> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  EXPECT_NEAR(out[0], static_cast<float>(in.size()), 1e-3f);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_double) {
  // Create data
  std::vector<double> in(1, 10);
  const int expected_sum = 10;
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<double> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  EXPECT_NEAR(out[0], expected_sum, 1e-6);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_int64_t) {
  // Create data
  std::vector<int64_t> in(75836, 1);
  std::vector<int64_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int64_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(static_cast<uint64_t>(out[0]), in.size());
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_uint8_t) {
  // Create data
  std::vector<uint8_t> in(255, 1);
  std::vector<uint8_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<uint8_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(static_cast<uint64_t>(out[0]), in.size());
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_empty) {
  // Create data
  std::vector<int32_t> in(1, 0);
  const int expected_sum = 0;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}
