// Copyright 2024 Nesterov Alexander
#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::pre_processing() {
  internal_order_test();
  input_ = std::vector<InOutType>(taskData->inputs_count[0]);
  auto tmp_ptr = reinterpret_cast<InOutType*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0;
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

template <class InOutType>
bool shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<InOutType>::post_processing() {
  internal_order_test();
  reinterpret_cast<InOutType*>(taskData->outputs[0])[0] = res;
  return true;
}
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<double>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<uint8_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int64_t>;
template class shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<float>;
