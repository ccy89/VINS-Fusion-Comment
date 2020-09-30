/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <pthread.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <cstdlib>
#include <unordered_map>

#include "../utility/tic_toc.h"
#include "../utility/utility.h"

const int NUM_THREADS = 4;

/**
 * @brief 边缘化残差块
 */
struct ResidualBlockInfo {
  /**
     * @brief 构造边缘化残差项
     * 
     * @param[in] _cost_function      残差计算方式
     * @param[in] _loss_function      鲁棒核函数
     * @param[in] _parameter_blocks   残差关联的所有变量
     * @param[in] _drop_set           待边缘化的优化变量在parameter_blocks 中的 id，例如{0, 1} 代表 _parameter_blocks[0] 和 _parameter_blocks[1] 是要边缘化掉的
     */
  ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
      : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

  void Evaluate();

  ceres::CostFunction *cost_function;       // 残差计算方式
  ceres::LossFunction *loss_function;       // 鲁棒核函数
  std::vector<double *> parameter_blocks;   // 所有优化变量数据在内存的地址
  std::vector<int> drop_set;                // 待边缘化的优化变量在 parameter_blocks 中的 id

  double **raw_jacobians;                   // 残差对每个变量Jacobian矩阵的地址
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;  // 残差对每个变量的jacobian矩阵
  Eigen::VectorXd residuals;                // 残差 IMU:15X1 视觉2X1

  int localSize(int size) {
    return size == 7 ? 6 : size;
  }
};

/**
 * @brief 多线程结构体
 */
struct ThreadsStruct {
  std::vector<ResidualBlockInfo *> sub_factors;   // 当前线程负责的所有残差块
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size;  //global size
  std::unordered_map<long, int> parameter_block_idx;   //local size
};

/**
 * @brief 边缘化信息体
 */
class MarginalizationInfo {
 public:
  MarginalizationInfo() { valid = true; };
  ~MarginalizationInfo();
  int localSize(int size) const;
  int globalSize(int size) const;
  void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);  // 添加边缘化残差项
  void preMarginalize();        // 边缘化预处理，计算每个残差块对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
  void marginalize();           // 执行边缘化：多线程构造先验项舒尔补AX=b的结构，计算舒尔补
  std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

  std::vector<ResidualBlockInfo *> factors;   // 需要边缘化的所有残差块
  int m, n;   // m为要边缘化的变量的总维度（local维度），n为要保留下来的变量维度（local维度）
  std::unordered_map<long, int> parameter_block_size;       // <变量内存地址，变量的globalSize> 
  int sum_block_size;   // 没有用到
  std::unordered_map<long, int> parameter_block_idx;        // <变量内存地址，变量在Hessian中的起始索引>
  std::unordered_map<long, double *> parameter_block_data;  // <变量内存地址，变量数据>

  // 进行边缘化之后保留下来的优化变量
  std::vector<int> keep_block_size;       // 每个非边缘化变量的 global维度
  std::vector<int> keep_block_idx;        // 每个非边缘化变量在Hessian中的起始索引 （包含边缘化变量在内）
  std::vector<double *> keep_block_data;  // 每个非边缘化变量在边缘化时刻的值(地址)

  Eigen::MatrixXd linearized_jacobians;   // 边缘化残差对变量的 Jacobian
  Eigen::VectorXd linearized_residuals;   // 边缘化残差
  const double eps = 1e-8;
  bool valid;
};


/**
 * @brief 先验因子，用于ceres中，继承自 ceres::CostFunction
 */
class MarginalizationFactor : public ceres::CostFunction {
 public:
  MarginalizationFactor(MarginalizationInfo *_marginalization_info);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  MarginalizationInfo *marginalization_info;
};
