/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../estimator/parameters.h"
#include "../utility/utility.h"

#include <ceres/ceres.h>
using namespace Eigen;

/**
 * @brief 两帧图像之间的 IMU预积分项
 */
class IntegrationBase {
 public:
  IntegrationBase() = delete;

  /**
   * @brief   imu 数据初始化，初始化高斯白噪声协方差矩阵
   * @param _acc_0    加速度值
   * @param _gyr_0    角速度值
   * @param _linearized_ba    加速度高斯白噪声标准差
   * @param _linearized_bg    角速度高斯白噪声标准差
   * 
  */
  IntegrationBase(const Eigen::Vector3d &_acc_0, 
                  const Eigen::Vector3d &_gyr_0,
                  const Eigen::Vector3d &_linearized_ba, 
                  const Eigen::Vector3d &_linearized_bg)
      : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0}, 
        linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg}, 
        jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()}, 
        sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()} {
    
    // imu数据 高斯白噪声协方差矩阵初始化
    noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
  }

  /**
   * @brief 传入当前的IMU数据，并进行中值积分
   * 
   * @param[in] dt    前后帧IMU时检差
   * @param[in] acc   当前加速度值
   * @param[in] gyr   当前角速度值
   */
  void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);
    propagate(dt, acc, gyr);    // 积分
  }

  /**
   * @brief 陀螺仪Bias更新后，需要根据新的bias重新计算预积分 
   * 
   * @param[in] _linearized_ba  新的ba
   * @param[in] _linearized_bg  新的bg
   */
  void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) {
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
      propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
  }


  /**
   * @brief 执行中值积分，并更新Jocabian 和 协方差矩阵
   * 
   * @param[in] _dt                     两帧时间差
   * @param[in] _acc_0                  上一帧加速度值 a_k
   * @param[in] _gyr_0                  上一帧角速度值 b_k
   * @param[in] _acc_1                  当前帧加速度值 a_k+1
   * @param[in] _gyr_1                  当前帧加速度值 b_k+1
   * @param[in] delta_p                 上一帧预积分项 ΔP
   * @param[in] delta_q                 上一帧预积分项 ΔQ
   * @param[in] delta_v                 上一帧预积分项 ΔV
   * @param[in] linearized_ba           上一帧 b_a
   * @param[in] linearized_bg           上一帧 b_g
   * @param[out] result_delta_p         当前帧预积分项 ΔP
   * @param[out] result_delta_q         当前帧预积分项 ΔQ
   * @param[out] result_delta_v         当前帧预积分项 ΔV
   * @param[out] result_linearized_ba   当前帧 b_a 
   * @param[out] result_linearized_bg   当前帧 b_g 
   * @param[in] update_jacobian         是否更新 Jacobian
   */
  void midPointIntegration(double _dt,
                           const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                           const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                           const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                           const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                           Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                           Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
    // IMU 数据中值积分
    Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);         // 计算上一帧 a_w0 = R_wb0 * (a_b0 - b_a) - g
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;      // ω_b = 0.5 * (ω_b0 + ω_b1) - b_g
    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);   // 中值积分项 R_wb1 = R_wb0 * ΔR_b0b1
    Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);  // 计算当前帧 a_w1 = R_wb1 * (a_b1 - b_a) - g
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);                  // 计算中值   a_w = 0.5 * (a_w0 + a_w1)
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;  // 中值积分项 P_wb1 = P_wb0 + V_w0 * dt + 0.5 * a_w * dt^2
    result_delta_v = delta_v + un_acc * _dt;    // 中值积分项 V_w1 = V_w0 + a_w * dt
    result_linearized_ba = linearized_ba;       // 短时间内传感器偏置不变
    result_linearized_bg = linearized_bg;       // 短时间内传感器偏置不变

    if (update_jacobian) {  // 是否更新协方差矩阵
      Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
      Vector3d a_0_x = _acc_0 - linearized_ba;
      Vector3d a_1_x = _acc_1 - linearized_ba;
      Matrix3d R_w_x, R_a_0_x, R_a_1_x;

      // 反对称矩阵
      R_w_x << 0, -w_x(2), w_x(1),
          w_x(2), 0, -w_x(0),
          -w_x(1), w_x(0), 0;
      R_a_0_x << 0, -a_0_x(2), a_0_x(1),
          a_0_x(2), 0, -a_0_x(0),
          -a_0_x(1), a_0_x(0), 0;
      R_a_1_x << 0, -a_1_x(2), a_1_x(1),
          a_1_x(2), 0, -a_1_x(0),
          -a_1_x(1), a_1_x(0), 0;

      // 计算协防差传递矩阵 F
      MatrixXd F = MatrixXd::Zero(15, 15);
      F.block<3, 3>(0, 0) = Matrix3d::Identity();
      F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
      F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
      F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
      F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
      F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
      F.block<3, 3>(6, 6) = Matrix3d::Identity();
      F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
      F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
      F.block<3, 3>(9, 9) = Matrix3d::Identity();
      F.block<3, 3>(12, 12) = Matrix3d::Identity();

      // 计算协方差传递矩阵 V
      MatrixXd V = MatrixXd::Zero(15, 18);
      V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
      V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
      V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
      V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
      V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

      jacobian = F * jacobian;  // 更新 IMU残差对偏置的Jocabian矩阵
      covariance = F * covariance * F.transpose() + V * noise * V.transpose();  // 更新 误差传递协方差
    }
  }

  /**
   * @brief 更新 预积分项
   * 
   * @param[in] _dt     IMU前后帧时间差
   * @param[in] _acc_1  当前帧加速度值
   * @param[in] _gyr_1  当前帧角速度值
   */
  void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    // 中值积分，并且更新 Jacobian 和 协方差
    midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                        linearized_ba, linearized_bg,
                        result_delta_p, result_delta_q, result_delta_v,
                        result_linearized_ba, result_linearized_bg, 1);

    //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
    //                    linearized_ba, linearized_bg);

    // 当前帧变成上一帧
    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
  }

  /**
   * @brief 计算当前预积分量的残差
   *        IMU的位姿是由相机位姿乘上外参获得的，用于和IMU自身预积分之间计算误差
   * 
   * @param[in] Pi        上一时刻IMU位姿 t_wb  
   * @param[in] Qi        上一时刻IMU位姿 R_wc
   * @param[in] Vi        上一时刻IMU速度 V_w
   * @param[in] Bai       上一时刻IMU偏置 ba
   * @param[in] Bgi       上一时刻IMU偏置 bg
   * @param[in] Pj        当前时刻IMU位姿 t_wb
   * @param[in] Qj        当前时刻IMU位姿 R_wc
   * @param[in] Vj        当前时刻IMU速度 V_w
   * @param[in] Baj       当前时刻IMU偏置 ba
   * @param[in] Bgj       当前时刻IMU偏置 bg
   * @return Eigen::Matrix<double, 15, 1>  残差 15维
   */
  Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, 
                                        const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, 
                                        const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
    Eigen::Matrix<double, 15, 1> residuals;

    Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbg = Bgi - linearized_bg;

    // 一阶泰勒展开近似计算含有偏置项的IMU预积分量
    Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    // 计算残差
    residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
  }

  double dt;                      // 当前帧和上一帧的时间差
  Eigen::Vector3d acc_0, gyr_0;   // 上一帧 IMU数据
  Eigen::Vector3d acc_1, gyr_1;   // 当前帧 IMU数据

  const Eigen::Vector3d linearized_acc, linearized_gyr;   // 备份当前预积分量第一帧IMU数据，重新预积分的时候有用
  Eigen::Vector3d linearized_ba, linearized_bg;           // 陀螺仪和加速度计的偏置

  Eigen::Matrix<double, 15, 15> jacobian;     // IMU残差对偏置的Jocabian矩阵
  Eigen::Matrix<double, 15, 15> covariance;   // IMU协方差传递矩阵 (是一个对称矩阵)
  Eigen::Matrix<double, 18, 18> noise;        // IMU高斯白噪声协方差矩阵

  double sum_dt;  // 总时差
  Eigen::Vector3d delta_p;    // 预积分量 ΔP
  Eigen::Quaterniond delta_q; // 预积分量 ΔQ
  Eigen::Vector3d delta_v;    // 预积分量 ΔV

  std::vector<double> dt_buf;             // 时间差序列
  std::vector<Eigen::Vector3d> acc_buf;   // 加速度值序列
  std::vector<Eigen::Vector3d> gyr_buf;   // 角速度值序列
};
