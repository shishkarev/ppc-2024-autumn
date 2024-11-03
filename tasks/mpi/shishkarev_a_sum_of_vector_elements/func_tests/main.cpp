// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_empty_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_1_size_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> ref_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_sum.data()));
    taskDataSeq->outputs_count.emplace_back(ref_sum.size());

    shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVectorSumSequential.validation());
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(ref_sum[0], global_sum[0]);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_100_size_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100;
    global_vec = shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_sum.data()));
    taskDataSeq->outputs_count.emplace_back(ref_sum.size());

    // Create Task
    shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVectorSumSequential.validation());
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(ref_sum[0], global_sum[0]);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_10000_size_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10000;
    global_vec = shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_sum.data()));
    taskDataSeq->outputs_count.emplace_back(ref_sum.size());

    // Create Task
    shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVectorSumSequential.validation());
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(ref_sum[0], global_sum[0]);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_1000000_size_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000000;
    global_vec = shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_sum.data()));
    taskDataSeq->outputs_count.emplace_back(ref_sum.size());

    // Create Task
    shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVectorSumSequential.validation());
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(ref_sum[0], global_sum[0]);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_0_size_sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 0);
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel.validation());
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> ref_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_sum.data()));
    taskDataSeq->outputs_count.emplace_back(ref_sum.size());

    shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVectorSumSequential.validation());
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(ref_sum[0], global_sum[0]);
  }
}
