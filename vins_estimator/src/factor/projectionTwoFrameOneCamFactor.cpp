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

#include "projectionTwoFrameOneCamFactor.h"

Eigen::Matrix2d ProjectionTwoFrameOneCamFactor::sqrt_info;
double ProjectionTwoFrameOneCamFactor::sum_t;

/**
 * @brief 构造含有时间戳同步的重投影误差
 * 
 * @param[in] _pts_i        特征点在 cam_i 左目的归一化平面上的坐标
 * @param[in] _pts_j        特征点在 cam_j 左目的归一化平面上的坐标
 * @param[in] _velocity_i   特征点在 cam_i 左目的像素运动速度
 * @param[in] _velocity_j   特征点在 cam_j 左目的像素运动速度 
 * @param[in] _td_i         cam_i 和 imu_i 之间的时间戳之差
 * @param[in] _td_j         cam_j 和 imu_j 之间的时间戳之差
 */
ProjectionTwoFrameOneCamFactor::ProjectionTwoFrameOneCamFactor(
    const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
    const double _td_i, const double _td_j) : pts_i(_pts_i), pts_j(_pts_j), td_i(_td_i), td_j(_td_j) {
  // 赋值
  velocity_i.x() = _velocity_i.x();
  velocity_i.y() = _velocity_i.y();
  velocity_i.z() = 0;
  velocity_j.x() = _velocity_j.x();
  velocity_j.y() = _velocity_j.y();
  velocity_j.z() = 0;

#ifdef UNIT_SPHERE_ERROR
  Eigen::Vector3d b1, b2;
  Eigen::Vector3d a = pts_j.normalized();
  Eigen::Vector3d tmp(0, 0, 1);
  if (a == tmp)
    tmp << 1, 0, 0;
  b1 = (tmp - a * (a.transpose() * tmp)).normalized();
  b2 = a.cross(b1);
  tangent_base.block<1, 3>(0, 0) = b1.transpose();
  tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};


/**
 * @brief  计算 重投影误差因子的 残差 和 Jacobian
 * 
 * @param[in] parameters    parameters[0~4]分别对应了4组优化变量的参数块
 * @param[out] residuals    重投影残差
 * @param[out] jacobians    残差对优化变量的Jacobian
 */
bool ProjectionTwoFrameOneCamFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  TicToc tic_toc;
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);                       // t_wbi
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);  // R_wbi  

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);                       // t_wbj
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);  // R_wbj

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);                      // t_bc
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); // R_bc

  double inv_dep_i = parameters[3][0];    // 逆深度 inv_dep

  double td = parameters[4][0];           // 相机和IMU时间戳之差 td

  // 假设像素匀速运动，计算考虑同步时间差的特征归一化坐标
  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i - (td - td_i) * velocity_i;
  pts_j_td = pts_j - (td - td_j) * velocity_j;

  // 获得 特征点在 cam_i 坐标系下的坐标 P_ci
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  // 获得 特征点在 imu_i 坐标系下的坐标 P_bi
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;               // P_bi = R_bc * P_ci + t_bc
  // 获得 特征点在的世界坐标i P_w 
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;                        // P_w = R_wb * P_bi + t_wb 
  // 获得 特征点在 imu_j 坐标系下的坐标 P_bj
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);            // P_bj = R_wbj^T * P_w - R_wbj^T * t_wbj = R_wbj^T * (P_w - t_wbj)
  // 获得 特征点在 cam_j 坐标系下的坐标 P_cj
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);   // P_cj =  R_bc^T * P_bj - R_bc^T * t_bc = R_bc^T * (P_bj - t_bc)  估计值


  Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();   // 估计 - 观测
#endif

  // 由于优化最后求解的方程 b = -J^T * Σ * r    
  //                    A = J^T * Σ * J        Σ是信息矩阵
  // 由于 ceres不支持单独设置信息矩阵，因此使用LLT分解信息矩阵,获得 sqrt(Σ)^T，J 和 r 都要乘上 sqrt(Σ)^T：
  // r' = sqrt(Σ)^T * r
  // J' = sqrt(Σ)^T * J
  // 这样在计算后端的时候   A = J'^T * J' = J^T * sqrt(Σ) * sqrt(Σ)^T * J = J^T * Σ * J
  //                    b = -J'^T * r' = -J^T * sqrt(Σ) * sqrt(Σ)^T * r = -J^T * Σ * r 

  residual = sqrt_info * residual;

  if (jacobians) {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
    double norm = pts_camera_j.norm();
    Eigen::Matrix3d norm_jaco;
    double x1, x2, x3;
    x1 = pts_camera_j(0);
    x2 = pts_camera_j(1);
    x3 = pts_camera_j(2);
    norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), -x1 * x2 / pow(norm, 3), -x1 * x3 / pow(norm, 3),
        -x1 * x2 / pow(norm, 3), 1.0 / norm - x2 * x2 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
        -x1 * x3 / pow(norm, 3), -x2 * x3 / pow(norm, 3), 1.0 / norm - x3 * x3 / pow(norm, 3);
    reduce = tangent_base * norm_jaco;
#else  // 相机模型
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
    reduce = sqrt_info * reduce;

    // 重投影误差对 pose_i 的 Jacobian
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

      Eigen::Matrix<double, 3, 6> jaco_i;
      jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
      jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

      jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
      jacobian_pose_i.rightCols<1>().setZero();
    }

    // 重投影误差对 pose_j 的 Jacobian
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

      Eigen::Matrix<double, 3, 6> jaco_j;
      jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
      jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

      jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
      jacobian_pose_j.rightCols<1>().setZero();
    }

    // 重投影误差对 T_bc 的 Jacobian
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
      Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
      jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                               Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
      jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose.rightCols<1>().setZero();
    }

    // 重投影误差对 逆深度 的 Jacobian
    if (jacobians[3]) {
      Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
      jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
    }

    // 重投影误差对 td 的 Jacobian
    if (jacobians[4]) {
      Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
      jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * velocity_j.head(2);
    }
  }
  sum_t += tic_toc.toc();

  return true;
}

void ProjectionTwoFrameOneCamFactor::check(double **parameters) {
  double *res = new double[2];
  double **jaco = new double *[5];
  jaco[0] = new double[2 * 7];
  jaco[1] = new double[2 * 7];
  jaco[2] = new double[2 * 7];
  jaco[3] = new double[2 * 1];
  jaco[4] = new double[2 * 1];
  Evaluate(parameters, res, jaco);
  puts("check begins");

  puts("my");

  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[4]) << std::endl
            << std::endl;

  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
  double inv_dep_i = parameters[3][0];
  double td = parameters[4][0];

  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i - (td - td_i) * velocity_i;
  pts_j_td = pts_j - (td - td_j) * velocity_j;
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
  Eigen::Vector2d residual;

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
  residual = sqrt_info * residual;

  puts("num");
  std::cout << residual.transpose() << std::endl;

  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 20> num_jacobian;
  for (int k = 0; k < 20; k++) {
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];
    double td = parameters[4][0];

    int a = k / 3, b = k % 3;
    Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

    if (a == 0)
      Pi += delta;
    else if (a == 1)
      Qi = Qi * Utility::deltaQ(delta);
    else if (a == 2)
      Pj += delta;
    else if (a == 3)
      Qj = Qj * Utility::deltaQ(delta);
    else if (a == 4)
      tic += delta;
    else if (a == 5)
      qic = qic * Utility::deltaQ(delta);
    else if (a == 6 && b == 0)
      inv_dep_i += delta.x();
    else if (a == 6 && b == 1)
      td += delta.y();

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Vector2d tmp_residual;

#ifdef UNIT_SPHERE_ERROR
    tmp_residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    tmp_residual = sqrt_info * tmp_residual;

    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << num_jacobian.block<2, 6>(0, 0) << std::endl;
  std::cout << num_jacobian.block<2, 6>(0, 6) << std::endl;
  std::cout << num_jacobian.block<2, 6>(0, 12) << std::endl;
  std::cout << num_jacobian.block<2, 1>(0, 18) << std::endl;
  std::cout << num_jacobian.block<2, 1>(0, 19) << std::endl;
}
