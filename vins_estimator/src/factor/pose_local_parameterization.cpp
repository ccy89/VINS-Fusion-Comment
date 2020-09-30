/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"

/**
 * @brief 四元数更新
 * 
 * @param[in] x             位姿原始值x_k
 * @param[in] delta         位姿增量Δx
 * @param[in] x_plus_delta  x_k+1 = x_k + Δx
 */
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> _p(x);        // 平移量
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3); // 旋转量

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));  // 将旋转增量转变为四元数的形式

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;                  // 更新平移
  q = (_q * dq).normalized();   // 更新旋转

  return true;
}


bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
  j.topRows<6>().setIdentity();
  j.bottomRows<1>().setZero();

  return true;
}
