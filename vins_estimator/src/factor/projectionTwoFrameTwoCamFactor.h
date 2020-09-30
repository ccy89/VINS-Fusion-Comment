/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <ros/assert.h>
#include <Eigen/Dense>
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

/**
 * @brief 计算特征在 相机i左目 和 相机j右目 上的重投影误差
 */
class ProjectionTwoFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1> {
 public:
  ProjectionTwoFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                 const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                 const double _td_i, const double _td_j);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector3d pts_i, pts_j;               // 特征点在相机归一化平面上的坐标
  Eigen::Vector3d velocity_i, velocity_j;     // 特征点的像素速度
  double td_i, td_j;                          // 相机和IMU时间差
  Eigen::Matrix<double, 2, 3> tangent_base;
  static Eigen::Matrix2d sqrt_info;           // 信息矩阵 LLT分解
  static double sum_t;
};
