// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

// Класс для представления матриц
class Matrix {
 public:
  std::vector<double> data;
  int rows, cols;

  Matrix() : rows(0), cols(0) {}
  Matrix(const std::vector<double>& data, int rows, int cols) : data(data), rows(rows), cols(cols) {}

  double& operator()(int i, int j) { return data[i * cols + j]; }
  const double& operator()(int i, int j) const { return data[i * cols + j]; }
};

class GaussianEliminationSequential : public ppc::core::Task {
 public:
  explicit GaussianEliminationSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix matA;
  std::vector<double> vecB;
  std::vector<double> result;
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq
