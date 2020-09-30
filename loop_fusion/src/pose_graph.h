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

#include <assert.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"
#include "ThirdParty/DVision/DVision.h"
#include "keyframe.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph {
 public:
  PoseGraph();
  ~PoseGraph();
  void registerPub(ros::NodeHandle& n);
  void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
  void loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
  void loadVocabulary(std::string voc_path);
  void setIMUFlag(bool _use_imu);
  KeyFrame* getKeyFrame(int index);
  nav_msgs::Path path[10];
  nav_msgs::Path base_path;
  CameraPoseVisualization* posegraph_visualization;
  void savePoseGraph();
  void loadPoseGraph();
  void publish();

  // w是前端VIO的世界坐标系
  // w0是PoseGraph的世界坐标系
  // w1是不同轨迹统一的世界坐标系 (base_sequence 或者 first_sequence)

  // T_w0_w1
  Vector3d t_drift;
  double yaw_drift;
  Matrix3d r_drift;

  // w1坐标系(base_sequence 或者 first_sequence) 和 当前轨迹所在坐标系 之间的变换矩阵  
  Vector3d w_t_vio;  // t_w0_vio
  Matrix3d w_r_vio;  // R_w0_vio

 private:
  int detectLoop(KeyFrame* keyframe, int frame_index);
  void addKeyFrameIntoVoc(KeyFrame* keyframe);
  void optimize4DoF();
  void optimize6DoF();
  void updatePath();
  list<KeyFrame*> keyframelist;  // 关键帧列表
  std::mutex m_keyframelist;
  std::mutex m_optimize_buf;
  std::mutex m_path;
  std::mutex m_drift;

  std::thread t_optimization;			// 闭环优化线程
  std::queue<int> optimize_buf;		// 闭环优化的关键帧队列

  int global_index;  // 当前关键帧在 PoseGraph 中的索引
  int sequence_cnt;  // 当前轨迹的编号
  vector<bool> sequence_loop;   // 轨迹对齐标记位 

  map<int, cv::Mat> image_pool;
  int earliest_loop_index;  // 整个 PoseGraph 最早的回环后选帧
  int base_sequence;
  bool use_imu;  // 是否使用IMU

  BriefDatabase db;      // Brief描述子数据库
  BriefVocabulary* voc;  // Brief描述子词典

  ros::Publisher pub_pg_path;
  ros::Publisher pub_base_path;
  ros::Publisher pub_pose_graph;
  ros::Publisher pub_path[10];
};


/**
 * @brief 四元数表示旋转，计算逆旋转矩阵
 * 对于四元数取逆矩阵，虚部不变，时部取负
 * 
 * @tparam T 
 * @param[in] q						旋转矩阵
 * @param[out] q_inverse	逆旋转矩阵		
 */
template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4]) {
  q_inverse[0] = q[0];
  q_inverse[1] = -q[1];
  q_inverse[2] = -q[2];
  q_inverse[3] = -q[3];
};


/**
 * @brief 归一化角度，使角度处于 -180 ~ +180 之间
 * 
 * @tparam T 
 * @param[in] angle_degrees		原始角度
 * @return T 									归一化以后的角度
 */
template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
    return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
    return angle_degrees + T(360.0);
  else
    return angle_degrees;
};

/**
 * @brief ceres 偏航角变量（1维）
 */
class AngleLocalParameterization {
 public:
  /**
   * @brief 重载变量更新操作
   * 
   * @tparam T 
   * @param[in] theta_radians               x_k       
   * @param[in] delta_theta_radians         Δx
   * @param[in] theta_radians_plus_delta    x_k+1 = x_k + Δx
   */
  template <typename T>
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  /**
   * @brief 创建自动求导的ceres变量
   */
  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
  }
};

/**
 * @brief 欧拉角转旋转矩阵，欧拉角顺序为 yaw -> pitch -> roll
 * 
 * @tparam T 
 * @param[in] yaw     偏航角
 * @param[in] pitch   俯仰角
 * @param[in] roll    翻滚角
 * @param[out] R      旋转矩阵
 */
template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9]) {
  // 角度转弧度
  T y = yaw / T(180.0) * T(M_PI);     
  T p = pitch / T(180.0) * T(M_PI);
  T r = roll / T(180.0) * T(M_PI);

  R[0] = cos(y) * cos(p);
  R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
  R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
  R[3] = sin(y) * cos(p);
  R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
  R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
  R[6] = -sin(p);
  R[7] = cos(p) * sin(r);
  R[8] = cos(p) * cos(r);
};

// 旋转矩阵转置
template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9]) {
  inv_R[0] = R[0];
  inv_R[1] = R[3];
  inv_R[2] = R[6];
  inv_R[3] = R[1];
  inv_R[4] = R[4];
  inv_R[5] = R[7];
  inv_R[6] = R[2];
  inv_R[7] = R[5];
  inv_R[8] = R[8];
};


/**
 * @brief 旋转一个向量
 * 
 * @tparam T      
 * @param[in] R     旋转矩阵
 * @param[in] t     向量 
 * @param[in] r_t   R * t
 */
template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3]) {
  r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
  r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
  r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

/**
 * @brief 4DoF的序列边误差，用来约束闭环边的
 *        该 CostFuction 计算在优化过程中，关键帧i、k之间的相对位姿 和 一开始观测到的相对位姿 之间的残差
 *        residuals 的定义 在第一轮迭代的时候为 0
 *        估计值和测量值 是同一个东西
 */
struct FourDOFError {
  /**
   * @brief 创建一个序列边，包含关键帧i、j的信息
   * 
   * @param[in] t_x             相对平移 t_bi_bj      (观测量)
   * @param[in] t_y             相对平移 t_bi_bj      (观测量)
   * @param[in] t_z             相对平移 t_bi_bj      (观测量)
   * @param[in] relative_yaw    相对偏航角 yaw_bi_bj  (观测量)
   * @param[in] pitch_i         关键帧i的roll 
   * @param[in] roll_i          关键帧i的pitch 
   * @return ceres::CostFunction* 
   */
  FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
      : t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {}

  /**
   * @brief 计算 观测值和估计值之间的残差
   * 
   * @tparam T 
   * @param[in] yaw_i       相机i的yaw    （优化变量）
   * @param[in] ti          相机i的t_w_bi （优化变量）
   * @param[in] yaw_j       相机j的yaw    （优化变量）
   * @param[in] tj          相机j的t_w_bj （优化变量）
   * @param[out] residuals  残差
   * @return true 
   * @return false 
   */
  template <typename T>
  bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T w_R_i[9];     // R_w_bi
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);   // 欧拉角到旋转矩阵

    T i_R_w[9];     // R_bi_w
    RotationMatrixTranspose(w_R_i, i_R_w);    // 旋转矩阵求转置

    T t_i_ij[3];    // t_bi_bj = R_bi_w * (t_w_bj - t_w_bi)
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    // 计算残差
    residuals[0] = (t_i_ij[0] - T(t_x));
    residuals[1] = (t_i_ij[1] - T(t_y));
    residuals[2] = (t_i_ij[2] - T(t_z));
    residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

    return true;
  }

  /**
   * @brief 创建一个自动求导的序列边
   * 
   * @param[in] t_x             相对平移 t_bi_bj      (观测量)
   * @param[in] t_y             相对平移 t_bi_bj      (观测量)
   * @param[in] t_z             相对平移 t_bi_bj      (观测量)
   * @param[in] relative_yaw    相对偏航角 yaw_bi_bj  (观测量)
   * @param[in] pitch_i         关键帧i的roll 
   * @param[in] roll_i          关键帧i的pitch 
   * @return ceres::CostFunction* 
   */
  static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                     const double relative_yaw, const double pitch_i, const double roll_i) {
    // 残差、相机i的yaw、相机i的t_w_bi、相机j的yaw、相机j的t_w_bi 维度分别维 4、1、3、1、3
    return (new ceres::AutoDiffCostFunction<
            FourDOFError, 4, 1, 3, 1, 3>(
        new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  // 观测量
  double t_x, t_y, t_z;     // 相对平移 t_bi_bj
  double relative_yaw;      // 相对yaw
  double pitch_i, roll_i;   // 关键帧i的roll 和 pitch，这两个量是可观的，不需要优化
};


/**
 * @brief 4DoF的闭环边，将关键帧和闭环帧在过去闭环时刻的相对位姿 和 当前时刻的相对位姿 构建误差
 */
struct FourDOFWeightError {
  /**
   * @brief 构造一个闭环帧
   * 
   * @param[in] t_x             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] t_y             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] t_z             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] relative_yaw    关键帧和闭环帧在闭环时刻的 yaw_loop_cur
   * @param[in] pitch_i         闭环帧在当前时刻的 pitch
   * @param[in] roll_i          闭环帧在当前时刻的 roll
   */
  FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
      : t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
    weight = 1;
  }

  /**
   * @brief 计算残差
   *        将关键帧和闭环帧在过去闭环时刻的相对位姿 和 当前时刻的相对位姿 构建误差
   * @tparam T 
   * @param[in] yaw_i       闭环帧在当前时刻的偏航角
   * @param[in] ti          闭环帧在当前时刻的平移量 t_wb
   * @param[in] yaw_j       关键帧在当前时刻的偏航角
   * @param[in] tj          关键帧在当前时刻的偏航角
   * @param[out] residuals  残差
   * @return true 
   * @return false 
   */
  template <typename T>
  bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    T w_R_i[9];   // R_w_loop
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);   // 欧拉角 转 旋转矩阵

    T i_R_w[9];   // R_loop_w
    RotationMatrixTranspose(w_R_i, i_R_w);    // 旋转矩阵转置

    T t_i_ij[3];  // t_loop_cur = R_loop_w * (t_w_cur - t_w_loop)
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);   // 旋转一个向量

    // 计算残差
    residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
    residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
    residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
    residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

    return true;
  }

  /**
   * @brief 创建一个自动求导的闭环边
   * 
   * @param[in] t_x             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] t_y             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] t_z             关键帧和闭环帧在闭环时刻的 t_loop_cur
   * @param[in] relative_yaw    关键帧和闭环帧在闭环时刻的 yaw_loop_cur
   * @param[in] pitch_i         闭环帧在当前时刻的 pitch
   * @param[in] roll_i          闭环帧在当前时刻的 roll
   * @return ceres::CostFunction* 
   */
  static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                     const double relative_yaw, const double pitch_i, const double roll_i) {
    return (new ceres::AutoDiffCostFunction<
            FourDOFWeightError, 4, 1, 3, 1, 3>(
        new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  // 观测值，关键帧和闭环帧在闭环时刻的相对位姿
  double t_x, t_y, t_z;       // t_loop_cur
  double relative_yaw;        // yaw_loop_cur

  double pitch_i, roll_i;     // 闭环帧在当前时刻的 pitch和roll，这两个量可观，因此不优化
  double weight;
};


/**
 * @brief 6DoF的序列边 或 闭环边
 */
struct RelativeRTError {
  /**
   * @brief 构造一个6DoF的序列边或闭环边
   * 
   * @param[in] t_x       两相机位姿的相对平移 tx (观测值)
   * @param[in] t_y       两相机位姿的相对平移 ty (观测值)
   * @param[in] t_z       两相机位姿的相对平移 tz (观测值)
   * @param[in] q_w       两相机位姿的相对旋转 qw (观测值)
   * @param[in] q_x       两相机位姿的相对旋转 qx (观测值)
   * @param[in] q_y       两相机位姿的相对旋转 qy (观测值)
   * @param[in] q_z       两相机位姿的相对旋转 qz (观测值)
   * @param[in] t_var     平移量的协方差
   * @param[in] q_var     旋转量的协方差
   */
  RelativeRTError(double t_x, double t_y, double t_z,
                  double q_w, double q_x, double q_y, double q_z,
                  double t_var, double q_var)
      : t_x(t_x), t_y(t_y), t_z(t_z), q_w(q_w), q_x(q_x), q_y(q_y), q_z(q_z), t_var(t_var), q_var(q_var) {}

  /**
   * @brief 计算残差
   *        估计当前关键帧i和关键帧j之间的相对位姿，和优化前的观测值构建误差       
   * 
   * @tparam T 
   * @param[in] w_q_i       关键帧i的 R_w_bi
   * @param[in] ti          关键帧i的 t_w_bi 
   * @param[in] w_q_j       关键帧j的 R_w_bj
   * @param[in] tj          关键帧j的 t_w_bj
   * @param[out] residuals  残差
   */
  template <typename T>
  bool operator()(const T* const w_q_i, const T* ti, const T* w_q_j, const T* tj, T* residuals) const {
    T t_w_ij[3];                
    t_w_ij[0] = tj[0] - ti[0];  
    t_w_ij[1] = tj[1] - ti[1];  
    t_w_ij[2] = tj[2] - ti[2];   

    T i_q_w[4];   // R_bi_w
    QuaternionInverse(w_q_i, i_q_w);      // 四元数取反

    T t_i_ij[3];
    ceres::QuaternionRotatePoint(i_q_w, t_w_ij, t_i_ij);    // t_bi_bj = R_bi_w * (R_w_bj - R_w_bi)
    // 计算平移量的残差
    residuals[0] = (t_i_ij[0] - T(t_x)) / T(t_var);
    residuals[1] = (t_i_ij[1] - T(t_y)) / T(t_var);
    residuals[2] = (t_i_ij[2] - T(t_z)) / T(t_var);

    T relative_q[4];
    relative_q[0] = T(q_w);
    relative_q[1] = T(q_x);
    relative_q[2] = T(q_y);
    relative_q[3] = T(q_z);

    T q_i_j[4];
    ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);  // R_bi_bj = R_bi_w * R_w_bj （估计值）

    T relative_q_inv[4];
    QuaternionInverse(relative_q, relative_q_inv);  // R_bj_bi (观测值)

    T error_q[4];
    ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);   // 计算观测值和估计值之间的误差

    // 计算 旋转量的误差
    residuals[3] = T(2) * error_q[1] / T(q_var);  
    residuals[4] = T(2) * error_q[2] / T(q_var);
    residuals[5] = T(2) * error_q[3] / T(q_var);

    return true;
  }

  /**
   * @brief 创建一个自动求导的 6DoF 序列边或者闭环边
   * 
   * @param[in] t_x       两相机位姿的相对平移 tx (观测值)
   * @param[in] t_y       两相机位姿的相对平移 ty (观测值)
   * @param[in] t_z       两相机位姿的相对平移 tz (观测值)
   * @param[in] q_w       两相机位姿的相对旋转 qw (观测值)
   * @param[in] q_x       两相机位姿的相对旋转 qx (观测值)
   * @param[in] q_y       两相机位姿的相对旋转 qy (观测值)
   * @param[in] q_z       两相机位姿的相对旋转 qz (观测值)
   * @param[in] t_var     平移量的方差
   * @param[in] q_var     旋转量的方差
   * @return ceres::CostFunction* 
   */
  static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                     const double q_w, const double q_x, const double q_y, const double q_z,
                                     const double t_var, const double q_var) {
    return (new ceres::AutoDiffCostFunction<
            RelativeRTError, 6, 4, 3, 4, 3>(
        new RelativeRTError(t_x, t_y, t_z, q_w, q_x, q_y, q_z, t_var, q_var)));
  }

  // 相对位姿的观测量
  double t_x, t_y, t_z, t_norm;     // t_norm 无用变量
  double q_w, q_x, q_y, q_z;
  double t_var, q_var;
};