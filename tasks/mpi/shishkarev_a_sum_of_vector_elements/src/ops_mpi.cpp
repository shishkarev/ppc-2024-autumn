// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

std::vector<int> shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(int vector_size) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0, 99);
  std::vector<int> random_vector(vector_size);
  std::generate(random_vector.begin(), random_vector.end(), [&]() { return distribution(generator); });
  return random_vector;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::pre_processing() {
  internal_order_test();

  // Копируем входные данные в локальный вектор
  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_vector.assign(input_ptr, input_ptr + taskData->inputs_count[0]);

  // Инициализация результата
  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::validation() {
  internal_order_test();

  // Убедимся, что выходной буфер имеет правильный размер
  return taskData->outputs_count[0] == 1;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::run() {
  internal_order_test();

  // Подсчитываем сумму элементов вектора
  result = std::accumulate(input_vector.cbegin(), input_vector.cend(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::post_processing() {
  internal_order_test();

  // Передаем результат в выходной буфер
  *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::pre_processing() {
  internal_order_test();

  // Определяем размер данных для разделения
  unsigned int delta = taskData->inputs_count[0] / mpi_comm.size();
  unsigned int remainder = taskData->inputs_count[0] % mpi_comm.size();

  if (mpi_comm.rank() == 0) {
    // Инициализируем входной вектор из данных задачи
    int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_vector.assign(input_ptr, input_ptr + taskData->inputs_count[0]);

    // Рассылаем части вектора другим процессам
    for (int proc = 1; proc < mpi_comm.size(); proc++) {
      unsigned int send_size = (proc == mpi_comm.size() - 1) ? delta + remainder : delta;
      mpi_comm.send(proc, 0, input_vector.data() + proc * delta, send_size);
    }
  }

  // Определяем размер локального вектора для текущего процесса
  unsigned int local_size = delta;
  if (mpi_comm.rank() == mpi_comm.size() - 1) {
    local_size += remainder;  // Для последнего процесса добавляем остаток
  }
  local_vector.resize(local_size);

  // Копируем данные в локальный вектор в зависимости от процесса
  if (mpi_comm.rank() == 0) {
    std::copy(input_vector.begin(), input_vector.begin() + delta, local_vector.begin());
  } else {
    mpi_comm.recv(0, 0, local_vector.data(), local_size);
  }

  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::validation() {
  internal_order_test();

  // Процесс 0 проверяет размер выходного буфера
  return mpi_comm.rank() != 0 || taskData->outputs_count[0] == 1;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::run() {
  internal_order_test();

  // Вычисляем локальную сумму
  int local_result = std::accumulate(local_vector.cbegin(), local_vector.cend(), 0);

  // Суммируем результаты всех процессов с использованием reduce
  boost::mpi::reduce(mpi_comm, local_result, result, std::plus<int>(), 0);  // Используем явный функтор std::plus<int>()
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::post_processing() {
  internal_order_test();

  // Процесс 0 записывает результат в выходной буфер
  if (mpi_comm.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  }
  return true;
}
