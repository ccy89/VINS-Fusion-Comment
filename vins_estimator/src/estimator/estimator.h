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
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

#include "../factor/imu_factor.h"
#include "../factor/marginalization_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/initial_sfm.h"
#include "../initial/solve_5pts.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "feature_manager.h"
#include "parameters.h"

class Estimator {
 public:
  Estimator();
  ~Estimator();
  void setParameter();

  // interface
  void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
  void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
  void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
  void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
  void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
  void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
  void processMeasurements();
  void changeSensorType(int use_imu, int use_stereo);

  // internal
  void clearState();
  bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
  void slideWindow();
  void slideWindowNew();
  void slideWindowOld();
  void optimization();
  void vector2double();
  void double2vector();
  bool failureDetection();
  bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                      vector<pair<double, Eigen::Vector3d>> &gyrVector);
  void getPoseInWorldFrame(Eigen::Matrix4d &T);
  void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
  void predictPtsInNextFrame();
  void outliersRejection(set<int> &removeIndex);
  double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                           Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                           double depth, Vector3d &uvi, Vector3d &uvj);
  void updateLatestStates();
  void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
  bool IMUAvailable(double t);
  void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

  enum SolverFlag {
    INITIAL,
    NON_LINEAR
  };

  enum MarginalizationFlag {
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
  };

  std::mutex mProcess;
  std::mutex mBuf;
  std::mutex mPropagate;
  queue<pair<double, Eigen::Vector3d>> accBuf;  // 加速度 队列  <imu时间戳，加速度值>
  queue<pair<double, Eigen::Vector3d>> gyrBuf;  // 角速度 队列  <imu时间戳，角速度值>
  // 特征点观测队列 <图像时间戳, <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点像素速度> > > >
  queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> featureBuf; 
  double prevTime, curTime;
  bool openExEstimation;

  std::thread trackThread;
  std::thread processThread;

  FeatureTracker featureTracker;    // 特征跟踪器

  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d g;   // 重力向量

  // IMU和相机之间的外参
  Matrix3d ric[2];    // R_bc
  Vector3d tic[2];    // t_bc

  Vector3d Ps[(WINDOW_SIZE + 1)];   // 滑窗内所有相机的 t_wc
  Vector3d Vs[(WINDOW_SIZE + 1)];   // 滑窗内所有相机的 V_w
  Matrix3d Rs[(WINDOW_SIZE + 1)];   // 滑窗内所有相机的 R_wc
  Vector3d Bas[(WINDOW_SIZE + 1)];  // 滑窗内所有相机对应的IMU ba
  Vector3d Bgs[(WINDOW_SIZE + 1)];  // 滑窗内所有相机对应的IMU bg
  double td;

  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  double Headers[(WINDOW_SIZE + 1)];

  IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
  Vector3d acc_0, gyr_0;

  vector<double> dt_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

  int frame_count;
  int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
  int inputImageCnt;

  FeatureManager f_manager;     // 特征管理器
  MotionEstimator m_estimator;
  InitialEXRotation initial_ex_rotation;

  bool first_imu;
  bool is_valid, is_key;
  bool failure_occur;

  vector<Vector3d> point_cloud;
  vector<Vector3d> margin_cloud;
  vector<Vector3d> key_poses;
  double initial_timestamp;

  // 以下变量用于ceres
  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];             // 滑窗中相机的位姿 T
  double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];   // 滑窗中相机的位姿 V
  double para_Feature[NUM_OF_F][SIZE_FEATURE];              // 滑窗中所有特征点的逆深度 
  double para_Ex_Pose[2][SIZE_POSE];                        // 相机和IMU之间的外参
  double para_Retrive_Pose[SIZE_POSE];                      
  double para_Td[1][1];                                     // 相机和IMU时间戳同步差
  double para_Tr[1][1];

  int loop_window_index;

  MarginalizationInfo *last_marginalization_info;
  vector<double *> last_marginalization_parameter_blocks;

  map<double, ImageFrame> all_image_frame;      // 图像帧 map 
  IntegrationBase *tmp_pre_integration;         // 

  Eigen::Vector3d initP;
  Eigen::Matrix3d initR;

  double latest_time;
  Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
  Eigen::Quaterniond latest_Q;

  bool initFirstPoseFlag;
  bool initThreadFlag;
};
