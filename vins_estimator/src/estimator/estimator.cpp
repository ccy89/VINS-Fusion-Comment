/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

/**
 * @brief Estimator 构造函数
 */
Estimator::Estimator() : f_manager{Rs} {
  ROS_INFO("init begins");
  initThreadFlag = false;   // slam初始化标志
  clearState();     // 初始化 Estimator数据
}

Estimator::~Estimator() {
  if (MULTIPLE_THREAD) {
    processThread.join();
    printf("join thread \n");
  }
}

void Estimator::clearState() {
  // 清空数据队列
  mProcess.lock();
  while (!accBuf.empty())       // 清空 加速度数据队列
    accBuf.pop();
  while (!gyrBuf.empty())       // 清空 角速度数据队列
    gyrBuf.pop();
  while (!featureBuf.empty())   // 清空 特征观测队列
    featureBuf.pop();

  prevTime = -1;
  curTime = 0;
  openExEstimation = 0;
  initP = Eigen::Vector3d(0, 0, 0);
  initR = Eigen::Matrix3d::Identity();
  inputImageCnt = 0;
  initFirstPoseFlag = false;
  
  // 清空滑动窗口内的数据
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr) {
      delete pre_integrations[i];
    }
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {  // 初始化 外参  R_ic 和 t_ic 为0
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  first_imu = false,
  sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  solver_flag = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();

  if (tmp_pre_integration != nullptr)
    delete tmp_pre_integration;
  if (last_marginalization_info != nullptr)
    delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();   // 初始化 特征管理器

  failure_occur = 0;

  mProcess.unlock();
}

void Estimator::setParameter() {
  mProcess.lock();
  for (int i = 0; i < NUM_OF_CAM; i++) {  // 设置外参 R_ic 和 t_ic
    tic[i] = TIC[i];
    ric[i] = RIC[i];
    cout << " exitrinsic cam " << i << endl
         << ric[i] << endl
         << tic[i].transpose() << endl;
  }
  f_manager.setRic(ric);  // 将外参传入特征管理器

  // 视觉观测信息矩阵，和相机双目基线成正比关系
  ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();  
  ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); 
  ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); 
  
  td = TD;  // 相机和imu时间差
  g = G;    // 重力方向
  cout << "set g " << g.transpose() << endl;
  featureTracker.readIntrinsicParameter(CAM_NAMES);   // 特征跟踪器获取 相机畸变参数

  std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';

  if (MULTIPLE_THREAD && !initThreadFlag) {
    initThreadFlag = true;
    processThread = std::thread(&Estimator::processMeasurements, this); // 程序主循环
  }
  mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo) {
  bool restart = false;
  mProcess.lock();
  if (!use_imu && !use_stereo)
    printf("at least use two sensors! \n");
  else {
    if (USE_IMU != use_imu) {
      USE_IMU = use_imu;
      if (USE_IMU) {
        // reuse imu; restart system
        restart = true;
      } else {
        if (last_marginalization_info != nullptr)
          delete last_marginalization_info;

        tmp_pre_integration = nullptr;
        last_marginalization_info = nullptr;
        last_marginalization_parameter_blocks.clear();
      }
    }

    STEREO = use_stereo;
    printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
  }
  mProcess.unlock();
  if (restart) {
    clearState();
    setParameter();
  }
}


/**
 * @brief Estimator 输入图像数据，并提取图像特征
 * 
 * @param[in] t     图像时间戳
 * @param[in] _img  左目图像    
 * @param[in] _img1 右目图像
 */
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1) {
  inputImageCnt++;
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  TicToc featureTrackerTime;

  if (_img1.empty())
    featureFrame = featureTracker.trackImage(t, _img);      // 单目跟踪
  else
    featureFrame = featureTracker.trackImage(t, _img, _img1);   // 双目跟踪
  
  // std::cout << "featureTracker time: " << featureTrackerTime.toc() << std::endl;   // 输出 光流跟踪时间
  
  if (SHOW_TRACK) {
    cv::Mat imgTrack = featureTracker.getTrackImage();
    pubTrackImage(imgTrack, t);   // 显示 光流跟踪的结果
  }


  if (MULTIPLE_THREAD) {
    // 如果使用多线程，只有偶数帧图像提取到的特征点会参与后续处理
    // 奇数帧图像提取的特征只用于前后帧的光流跟踪
    if (inputImageCnt % 2 == 0) {   // 
      mBuf.lock();
      // 构建 特征点观测 队列
      // 成员数据结构：<图像时间戳, <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点像素速度> > > >
      featureBuf.push(make_pair(t, featureFrame));
      mBuf.unlock();
    }
  } else {
    // 如果不使用多线程，每一帧图像提取到的特征点会参与后续处理
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();
    TicToc processTime;
    processMeasurements();    // 处理图像和IMU数据，程序主循环
    printf("process time: %f\n", processTime.toc());
  }
}


/**
 * @brief estimator 接收 imu 数据
 * 
 * @param[in] t                   IMU时间戳
 * @param[in] linearAcceleration  IMU加速度
 * @param[in] angularVelocity     IMU角速度
 */
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity) {
  mBuf.lock(); 
  accBuf.push(make_pair(t, linearAcceleration));  // 存储 imu时间戳 加速度值 
  gyrBuf.push(make_pair(t, angularVelocity));     // 存储 imu时间戳 角速度值
  mBuf.unlock(); 

  if (solver_flag == NON_LINEAR) {  // 如果系统初始化完成
    mPropagate.lock();
    fastPredictIMU(t, linearAcceleration, angularVelocity);   // 快速预积分
    pubLatestOdometry(latest_P, latest_Q, latest_V, t);       // 直接pub预积分的位姿
    mPropagate.unlock();  
  }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame) {
  mBuf.lock();
  featureBuf.push(make_pair(t, featureFrame));
  mBuf.unlock();

  if (!MULTIPLE_THREAD)
    processMeasurements();
}

bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector) {
  if (accBuf.empty()) {
    printf("not receive imu\n");
    return false;
  }
  //printf("get imu from %f %f\n", t0, t1);
  //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
  if (t1 <= accBuf.back().first) {
    while (accBuf.front().first <= t0) {
      accBuf.pop();
      gyrBuf.pop();
    }
    while (accBuf.front().first < t1) {
      accVector.push_back(accBuf.front());
      accBuf.pop();
      gyrVector.push_back(gyrBuf.front());
      gyrBuf.pop();
    }
    accVector.push_back(accBuf.front());
    gyrVector.push_back(gyrBuf.front());
  } else {
    printf("wait for imu\n");
    return false;
  }
  return true;
}

/**
 * @brief 
 * 
 * @param[in] t
 * @return true 
 * @return false 
 */
bool Estimator::IMUAvailable(double t) {
  if (!accBuf.empty() && t <= accBuf.back().first)
    return true;
  else
    return false;
}

/**
 * @brief 处理图像和IMU数据，估计当前帧位姿，是整个程序的主循环
 */
void Estimator::processMeasurements() {
  while (1) {
    //printf("process measurments\n");
    pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;   // 特征观测
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;   // imu数据
    // 等待有图像完成特征提取
    if (!featureBuf.empty()) {
      feature = featureBuf.front();   // 获得 队列中最早的 图像的特征点
      curTime = feature.first + td;   // 获得 图像的时间戳，定义为当前帧时间戳
      while (1) {
        if ((!USE_IMU || IMUAvailable(curTime)))  // 如果不使用 IMU 或者 当前帧时间戳 <= 最新一帧的imu时间戳 (由于 imu频率远大于图像帧率)
          break;
        else {
          printf("wait for imu ... \n");    // 等待IMU数据
          if (!MULTIPLE_THREAD)
            return;
          std::chrono::milliseconds dura(5);
          std::this_thread::sleep_for(dura);
        }
      }

      mBuf.lock();
      if (USE_IMU) {
        // 获取 前后两帧图像之间的所有imu数据，存储到 accVector 和 gyrVector 之中
        getIMUInterval(prevTime, curTime, accVector, gyrVector);  
      }
        
      featureBuf.pop();   // 当前帧特征点从队列中移除
      mBuf.unlock();

      if (USE_IMU) {
        if (!initFirstPoseFlag)   // 如果imu未初始化
          initFirstIMUPose(accVector);  // 计算imu初始位姿
        
        // IMU预积分 
        for (size_t i = 0; i < accVector.size(); i++) {
          double dt;
          if (i == 0)
            dt = accVector[i].first - prevTime;
          else if (i == accVector.size() - 1)
            dt = curTime - accVector[i - 1].first;
          else
            dt = accVector[i].first - accVector[i - 1].first;
          processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // imu 中值积分
        }
      }
      mProcess.lock();
      processImage(feature.second, feature.first);    // 处理图像数据
      prevTime = curTime;

      printStatistics(*this, 0);

      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time(feature.first);

      pubOdometry(*this, header);
      pubKeyPoses(*this, header);
      pubCameraPose(*this, header);
      pubPointCloud(*this, header);
      pubKeyframe(*this);
      pubTF(*this, header);
      mProcess.unlock();
    }

    if (!MULTIPLE_THREAD)
      break;

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector) {
  printf("init first imu pose\n");
  initFirstPoseFlag = true;   // 修改标志位

  Eigen::Vector3d averAcc(0, 0, 0);
  int n = (int)accVector.size();    // 获得 imu数据个数
  for (size_t i = 0; i < accVector.size(); i++) {
    averAcc = averAcc + accVector[i].second;  // 累加imu数据
  }

  averAcc = averAcc / n;    // 计算 平均加速度
  printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());

  Matrix3d R0 = Utility::g2R(averAcc);    // 计算当前加速度计值和重力向量之间的旋转 R_wb
  // 下面两行应该是多余的，在 Utility::g2R() 函数里以及执行过了
  double yaw = Utility::R2ypr(R0).x();    // 获得 偏航角
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 设置偏航角为0，由于没有磁力计，偏航角不可观
  Rs[0] = R0;   // 设置 imu 初始位姿
  cout << "init R0 " << endl
       << Rs[0] << endl;

}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r) {
  Ps[0] = p;
  Rs[0] = r;
  initP = p;
  initR = r;
}

/**
 * @brief   IMU中值积分           
 *
 * @param[in]   t   当前帧IMU数据时间戳
 * @param[in]   dt  前后两帧IMU数据时间差
 * @param[in]   linear_acceleration 当前加速度值
 * @param[in]   angular_velocity    当前角速度值
 * 
 * @return  void
*/
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
  if (!first_imu) { // 第一帧imu数据，初始化imu 预积分
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  if (!pre_integrations[frame_count]) {   
    pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
  }
  if (frame_count != 0) {   // 只要不是第一帧，就进行中值积分

    // 这里的积分是两帧图像之间的增量
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);  // 传入IMU数据，进行中值积分
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);            // 传入IMU数据，进行中值积分

    dt_buf[frame_count].push_back(dt);                                    // 存储 dt
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);  // 存储 加速度值
    angular_velocity_buf[frame_count].push_back(angular_velocity);        // 存储 角速度值

    // 这里的积分是获得当前的位姿
    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;               // 计算上一帧 a_w = R_wb * (a_b - b_a) - g
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];    // 计算当前帧 ω_b = 0.5 * (ω_b0 + ω_b1) - b_g
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();       // 计算当前帧 中值积分项 R_wb = R_wb * ΔR
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g; // 计算当前帧 a_w = R_wb * (a_b - b_a) - g
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);  // 计算当前帧 a_w = 0.5 * (a_w0 + a_w1)
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;   // 计算当前帧 中值积分项 P_wb
    Vs[j] += dt * un_acc;                           // 计算当前 中值积分项 V_w
  }
  // 存储当前帧 IMU 数据，当前帧变成上一帧
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

/**
 * @brief 处理最新帧图像
 * Step 1: 判断次新帧是否为关键帧，决定边缘化方式
 * Step 2: 如果配置文件没有提供IMU和相机之间的外参，则标定该参数
 * Step 3: 如果系统还未初始化，则初始化系统
 * Step 4: 存视觉使用PnP求解当前帧位姿
 * Step 5: 三角化求解特征点在当前帧的深度
 * Step 6: 滑窗后端优化，执行边缘化操作，计算先验信息
 * Step 7: 计算优化后特征的的重投影误差，剔除outlier
 * Step 8: 对于单线程配置，剔除前端跟踪的outlier，并预测特征在下一帧的位置
 * Step 9: 判断系统是否出错，一旦检测到故障，系统将切换回初始化阶段
 * Step 10: 执行滑动窗口，丢弃边缘化帧的观测
 * Step 11: 删除跟踪失败的特征
 * Step 12: 更新最新帧的状态，用于rviz显示
 * 
 * @param[in] image   当前图像特征跟踪情况，数据结构为 <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点像素速度>>>
 * @param[in] header  当前帧图像时间戳
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header) {
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());

  // Step 1: 判断次新帧是否为关键帧，决定边缘化方式
  // 将当前帧图像 检测到的特征点 添加到 feature容器中，计算每一个点跟踪的次数，以及它的视差
  // 通过检测 上一帧和上上帧图像之间的视差 | 当前图像为是否滑动窗口起始帧2帧 来决定上一帧是否作为关键帧
  if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
    // 上一帧是关键帧，则后端优化时移除滑窗中的第一帧，当前帧插入滑窗末尾
    marginalization_flag = MARGIN_OLD;
  } else {
    // 上一帧不是关键帧，则后端优化时直接移除滑动窗口中的最后一帧，当前帧插入滑窗末尾
    marginalization_flag = MARGIN_SECOND_NEW;
  }

  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
  Headers[frame_count] = header;  // 记录当前帧图像的时间戳

  ImageFrame imageframe(image, header);   // 构造图像帧
  imageframe.pre_integration = tmp_pre_integration;       // 存储 预积分量
  all_image_frame.insert(make_pair(header, imageframe));  // 将当前图像帧存入 图像帧map
  tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};  // 更新 临时预积分初始值

  // Step 2: 如果配置文件没有提供IMU和相机之间的外参，则标定该参数
  if (ESTIMATE_EXTRINSIC == 2) {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0) {
      vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                       << calib_ric);
        ric[0] = calib_ric;
        RIC[0] = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  // Step 3: 如果系统还未初始化，则初始化系统
  if (solver_flag == INITIAL) {   // 如果还未初始化
    // 单目 + IMU 初始化
    if (!STEREO && USE_IMU) {   
      if (frame_count == WINDOW_SIZE) {
        bool result = false;
        if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
          result = initialStructure();
          initial_timestamp = header;
        }
        if (result) {
          optimization();
          updateLatestStates();
          solver_flag = NON_LINEAR;
          slideWindow();
          ROS_INFO("Initialization finish!");
        } else
          slideWindow();
      }
    }

    // 双目 + IMU 初始化
    if (STEREO && USE_IMU) {
      // Step 3.1: PnP 求解当前帧相机位姿 T_wc
      // 虽然 initFramePoseByPnP() 在 triangulate() 前面，但第一次计算时，由于没有任何三维深度信息，无法用 pnp 求解位姿，因此会先进行三角化计算
      // 一旦有了深度，当下一帧图像来到以后就可以利用 pnp 求解位姿
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);

      // Step 3.2: 三角化求解当前帧图像特征点的深度
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
      if (frame_count == WINDOW_SIZE) {   
        // 如果滑动窗口满了，对陀螺仪Bias 进行校正，并进行初始化
        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        // 遍历 图像帧map
        for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
          // 将当前帧的 T_wb 存入 当前的图像帧
          frame_it->second.R = Rs[i];
          frame_it->second.T = Ps[i];
          i++;
        }
        // Step 3.3: 校正陀螺仪的Bias，并更新预积分
        solveGyroscopeBias(all_image_frame, Bgs);
        for (int i = 0; i <= WINDOW_SIZE; i++) {
          pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]); // 由于Bias改变，更新预积分量
        }
        // Step 3.4: 后端优化
        optimization();

        // Step 3.5: 更新最新帧的状态，用于rviz显示
        updateLatestStates();
        solver_flag = NON_LINEAR;

        // Step 3.6: 滑动窗口
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    // 双目 初始化
    if (STEREO && !USE_IMU) {
      // Step 3.1: PnP 求解当前帧相机位姿 T_wc
      // 虽然 initFramePoseByPnP() 在 triangulate() 前面，但第一次计算时，由于没有任何三维深度信息，无法用 pnp 求解位姿，因此会先进行三角化计算
      // 一旦有了深度，当下一帧图像来到以后就可以利用 pnp 求解位姿
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);

      // Step 3.2: 三角化求解当前帧图像特征点的深度
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

      // Step 3.3: 优化位姿 
      optimization();

      if (frame_count == WINDOW_SIZE) {
        // Step 3.4: 后端优化
        optimization();

        // Step 3.5: 更新最新帧的状态，用于rviz显示
        updateLatestStates();
        solver_flag = NON_LINEAR;

        // Step 3.6: 滑动窗口
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    if (frame_count < WINDOW_SIZE) {
      frame_count++;
      int prev_frame = frame_count - 1;
      Ps[frame_count] = Ps[prev_frame];
      Vs[frame_count] = Vs[prev_frame];
      Rs[frame_count] = Rs[prev_frame];
      Bas[frame_count] = Bas[prev_frame];
      Bgs[frame_count] = Bgs[prev_frame];
    }

  } else {      
    // 已经完成初始化

    TicToc t_solve;
    if (!USE_IMU) { 
      // Step 4: 存视觉使用 pnp 算法估计当前帧位姿
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);  
    } 

    // Step 5: 三角化求解特征点在当前帧的深度
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);   // 线性三角法 求解当前帧图像特征点的深度

    // Step 6: 滑窗后端优化，执行边缘化操作，计算先验信息
    optimization();   // 后端优化
    
    // Step 7: 计算优化后特征的的重投影误差，剔除outlier
    set<int> removeIndex;
    outliersRejection(removeIndex);         // 检测重投影误差，获得优化后的Outliers
    f_manager.removeOutlier(removeIndex);   // 从特征观测队列中删除优化后的Outliers

    // Step 8: 对于单线程配置，剔除前端跟踪的outlier，并预测特征在下一帧的位置
    if (!MULTIPLE_THREAD) {
      // 将outlier的特征在前端跟踪中剔除
      featureTracker.removeOutliers(removeIndex);
      predictPtsInNextFrame();  // 基于恒速模型，预测路标点在下一时刻左图中的坐标
    }

    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    // Step 9: 判断系统是否出错，一旦检测到故障，系统将切换回初始化阶段
    if (failureDetection()) {
      ROS_WARN("failure detection!");
      failure_occur = 1;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    // Step 10: 执行滑动窗口，丢弃边缘化帧的观测
    slideWindow();

    // Step 11: 删除跟踪失败的特征
    f_manager.removeFailures();   // 
    
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.push_back(Ps[i]);

    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];

    // Step 12: 更新最新帧的状态，用于rviz显示
    updateLatestStates();
  }
}


bool Estimator::initialStructure() {
  TicToc t_sfm;
  //check imu observibility
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
      //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    //ROS_WARN("IMU variation %f!", var);
    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");
      //return false;
    }
  }
  // global sfm
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;
  for (auto &it_per_id : f_manager.feature) {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)) {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }
  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l,
                     relative_R, relative_T,
                     sfm_f, sfm_tracked_points)) {
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  //solve pnp for all frame
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers[i]) {
      frame_it->second.is_key_frame = true;
      frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
      frame_it->second.T = T[i];
      i++;
      continue;
    }
    if ((frame_it->first) > Headers[i]) {
      i++;
    }
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto &id_pts : frame_it->second.points) {
      int feature_id = id_pts.first;
      for (auto &i_p : id_pts.second) {
        it = sfm_tracked_points.find(feature_id);
        if (it != sfm_tracked_points.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * RIC[0].transpose();
    frame_it->second.T = T_pnp;
  }
  if (visualInitialAlign())
    return true;
  else {
    ROS_INFO("misalign visual structure with IMU");
    return false;
  }
}

bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  //solve scale
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
  if (!result) {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // change state
  for (int i = 0; i <= frame_count; i++) {
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    Vector3d Pi = all_image_frame[Headers[i]].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  for (int i = frame_count; i >= 0; i--)
    Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  Matrix3d R0 = Utility::g2R(g);
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g = R0 * g;
  //Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  f_manager.clearDepth();
  f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

  return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
  // find previous frame which contians enough correspondance and parallex with newest frame
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);
    if (corres.size() > 20) {
      double sum_parallax = 0;
      double average_parallax;
      for (int j = 0; j < int(corres.size()); j++) {
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());
      if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        l = i;
        ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief 由于 ceres 使用数值，因此要将 数组vector 转换成 double数组
 *        Ps、Rs 转变成 para_Pose
 *        Vs、Bas、Bgs 转变成 para_SpeedBias
 *        R_bc 转变为 para_Ex_Pose
 *        逆深度 转变为 para_Feature
 *        td 转变为 para_Td
 */
void Estimator::vector2double() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    if (USE_IMU) {
      para_SpeedBias[i][0] = Vs[i].x();
      para_SpeedBias[i][1] = Vs[i].y();
      para_SpeedBias[i][2] = Vs[i].z();

      para_SpeedBias[i][3] = Bas[i].x();
      para_SpeedBias[i][4] = Bas[i].y();
      para_SpeedBias[i][5] = Bas[i].z();

      para_SpeedBias[i][6] = Bgs[i].x();
      para_SpeedBias[i][7] = Bgs[i].y();
      para_SpeedBias[i][8] = Bgs[i].z();
    }
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);

  para_Td[0][0] = td;
}

/**
 * @brief 从ceres的结果中恢复优化变量，并且保证滑窗中第一帧相机的偏航角不变
 */
void Estimator::double2vector() {
  // Step 1: 记录 滑窗中第一帧位姿优化前的值
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]); // 优化前的欧拉角
  Vector3d origin_P0 = Ps[0];

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = 0;
  }

  // Step 2: 计算 滑窗中第一帧相机位姿在优化前后的偏航角变化量
  if (USE_IMU) {
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());      // 优化后的欧拉角
    double y_diff = origin_R0.x() - origin_R00.x();   // 偏航角之差
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));   // 偏航角变化量对应的旋转矩阵
    // 如果欧拉角存在歧义
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
      ROS_DEBUG("euler singular point!");
      rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                     para_Pose[0][3],
                                     para_Pose[0][4],
                                     para_Pose[0][5])
                             .toRotationMatrix()
                             .transpose();
    }

    // Step 3: 从ceres中恢复优化变量，并且保证滑窗中第一帧相机的偏航角不变
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

      Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                  para_Pose[i][1] - para_Pose[0][1],
                                  para_Pose[i][2] - para_Pose[0][2]) +
              origin_P0;

      Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                  para_SpeedBias[i][1],
                                  para_SpeedBias[i][2]);

      Bas[i] = Vector3d(para_SpeedBias[i][3],
                        para_SpeedBias[i][4],
                        para_SpeedBias[i][5]);

      Bgs[i] = Vector3d(para_SpeedBias[i][6],
                        para_SpeedBias[i][7],
                        para_SpeedBias[i][8]);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

      Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
    }
  }

  if (USE_IMU) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      tic[i] = Vector3d(para_Ex_Pose[i][0],
                        para_Ex_Pose[i][1],
                        para_Ex_Pose[i][2]);
      ric[i] = Quaterniond(para_Ex_Pose[i][6],
                           para_Ex_Pose[i][3],
                           para_Ex_Pose[i][4],
                           para_Ex_Pose[i][5])
                   .normalized()
                   .toRotationMatrix();
    }
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);

  if (USE_IMU)
    td = para_Td[0][0];
}

/**
 * @brief 检测系统是否发生错误
 */
bool Estimator::failureDetection() {
  return false;
  if (f_manager.last_track_num < 2) {   // 最新帧跟踪到的特征太少
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    //return true;
  }

  if (Bas[WINDOW_SIZE].norm() > 2.5) {  // 加速度计的bias太大了
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) {  // 陀螺仪的bias太大了
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    //ROS_INFO(" big translation");
    //return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    //ROS_INFO(" big z translation");
    //return true;
  }
  Matrix3d tmp_R = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    ROS_INFO(" big delta_angle ");
    //return true;
  }
  return false;
}


/**
 * @brief 基于滑动窗口紧耦合的非线性优化，残差项的构造和求解  
 *        添加要优化的变量 (p, v, q, ba, bg) 一共15个自由度，IMU 的外参 R_bc 也可以加进来
 *        添加残差，残差项分为4块 先验残差 + IMU残差 + 视觉残差
 *        根据倒数第二帧是不是关键帧确定边缘化的策略     
 */
void Estimator::optimization() {
  TicToc t_whole, t_prepare;
  vector2double();    // 将优化变量转变为数组的形式

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  //loss_function = NULL;
  loss_function = new ceres::HuberLoss(1.0);    // 定义损失函数的鲁棒核
  //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);

  // 遍历滑窗内每一帧
  for (int i = 0; i < frame_count + 1; i++) { 
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    if (USE_IMU)  // 如果使用IMU，则优化 Vs，Bgs，Bas
      problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }

  if (!USE_IMU)
    problem.SetParameterBlockConstant(para_Pose[0]);  // 存视觉VO，优化过程中 固定滑窗内第一帧的位姿

  for (int i = 0; i < NUM_OF_CAM; i++) {  // 将 R_bc 添加进优化
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

    if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation) {
      openExEstimation = 1;   // 优化 R_bc
    } else {
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);   // 优化过程中固定 R_bc
    }
  }

  // 将 td 添加进优化 
  problem.AddParameterBlock(para_Td[0], 1);   

  if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
    problem.SetParameterBlockConstant(para_Td[0]);  // 优化过程中 固定 td

  // 添加边缘化残差
  if (last_marginalization_info && last_marginalization_info->valid) {
    // construct new marginlization_factor
    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, NULL,
                             last_marginalization_parameter_blocks);
  }

  // 添加 IMU 残差项
  if (USE_IMU) {
    for (int i = 0; i < frame_count; i++) {
      int j = i + 1;
      if (pre_integrations[j]->sum_dt > 10.0)
        continue;
      IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
      problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
  }

  // 添加视觉残差项
  int f_m_cnt = 0;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature) {   // 遍历所有的特征点
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)   // 如果当前特征点被观测到的次数 < 4，则直接跳过 
      continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point;    // 获得 当前特征点在第一次被观测到时的归一化坐标

    for (auto &it_per_frame : it_per_id.feature_per_frame) {  // 遍历观测到当前特征点的所有帧
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
        problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
      }

      if (STEREO && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
        } else {
          ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
        }
      }
      f_m_cnt++;
    }
  }

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  //printf("prepare for ceres: %f \n", t_prepare.toc());

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;    // 使用稠密矩阵
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG; // 使用 dogleg算法 
  options.max_num_iterations = NUM_ITERATIONS;        // 设置最大迭代次数
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;
  if (marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);    // 求解问题
  //cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  //printf("solver costs: %f \n", t_solver.toc());

  // 从ceres优化结果中恢复出变量
  double2vector();
  //printf("frame_count: %d \n", frame_count);

  if (frame_count < WINDOW_SIZE)  // 滑动窗口还没有满，则不进行边缘化操作
    return;

  /************************边缘化处理*****************************/
  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD) {   // 如果 边缘化掉最老帧
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    vector2double();
    // 1. 将上一次先验残差项传递给marginalization_info
    if (last_marginalization_info && last_marginalization_info->valid) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                     last_marginalization_parameter_blocks,
                                                                     drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    // 2. 将第1帧和第2帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
    if (USE_IMU) {
      if (pre_integrations[1]->sum_dt < 10.0) {
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                       vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                       vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // 3. 最后将第一次观测为滑窗中第1帧的路标点 以及 滑窗中和第1帧共视该路标点的相机添加进marginalization_info中
    {
      int feature_index = -1;
      // 遍历滑窗内所有跟踪到的特征
      for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)   // 如果不是长期跟踪到的点 不考虑
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i != 0)   // 如果当前特征第一个观察帧不是第1帧就不进行考虑
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 获得当前特征在第1帧相机归一化平面的坐标
        // 遍历滑窗内观测到当前特征的每一帧
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;
          if (imu_i != imu_j) {
            Vector3d pts_j = it_per_frame.point;   // 获得 当前特征 在 相机j 归一化平面的坐标
            ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                      it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                           vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          if (STEREO && it_per_frame.is_stereo) {
            Vector3d pts_j_right = it_per_frame.pointRight;
            if (imu_i != imu_j) {
              ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
              ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                             vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                             vector<int>{0, 4});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            } else {
              ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
              ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                             vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                             vector<int>{2});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            }
          }
        }
      }
    }

    // 4. 计算每个残差块对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    // 5. 执行边缘化：多线程构造先验项舒尔补AX=b的结构，计算舒尔补
    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    // 6. 调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
      if (USE_IMU)
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
      addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

    addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info)
      delete last_marginalization_info;
    last_marginalization_info = marginalization_info;         // 记录当前先验信息
    last_marginalization_parameter_blocks = parameter_blocks; // 记录当前先验信息中非边缘化变量的地址

  } else {
    // 如果上一帧不是关键帧，边缘化掉次新帧
    // 存在先验边缘化信息时才进行次新帧边缘化；否则仅仅通过slidewindow 丢弃次新帧
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) {
      MarginalizationInfo *marginalization_info = new MarginalizationInfo();    // 构造新的 边缘化信息体
      vector2double();

      // 设置从上一次的先验信息中边缘化次新帧的位姿信息
      if (last_marginalization_info && last_marginalization_info->valid) {
        vector<int> drop_set;   // 记录需要丢弃的变量在last_marginalization_parameter_blocks中的索引
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);   // 使用上一次先验残差 构建 当前的先验残差
        // 从上一次的先验残差中设置边缘化次新帧的位姿信息
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                       last_marginalization_parameter_blocks,
                                                                       drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      // 2. 计算每个残差块对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      // 3. 执行边缘化：多线程构建Hessian矩阵，计算舒尔补
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      // 4. 调整参数块在下一次窗口中对应的位置（去掉次新帧）
      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)   // 上一帧被丢弃
          continue;
        else if (i == WINDOW_SIZE) {  // 当前帧覆盖上一帧
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        } else {    // 其他帧不动
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
        }
      }
      for (int i = 0; i < NUM_OF_CAM; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

      addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

      vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info)
        delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
  //printf("whole time for ceres: %f \n", t_whole.toc());
}

/**
 * @brief 实现滑动窗口
 * 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
 * 如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 */
void Estimator::slideWindow() {
  TicToc t_margin;
  if (marginalization_flag == MARGIN_OLD) {   // 如果边缘化最老帧
    // 备份最老帧的数据
    double t_0 = Headers[0];
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count == WINDOW_SIZE) {
      // 将前后帧数据交换，最终结果为 1 2 3 4 5 6 7 8 9 10 0
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Headers[i] = Headers[i + 1];
        Rs[i].swap(Rs[i + 1]);
        Ps[i].swap(Ps[i + 1]);
        if (USE_IMU) {
          std::swap(pre_integrations[i], pre_integrations[i + 1]);

          dt_buf[i].swap(dt_buf[i + 1]);
          linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
          angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

          Vs[i].swap(Vs[i + 1]);
          Bas[i].swap(Bas[i + 1]);
          Bgs[i].swap(Bgs[i + 1]);
        }
      }
      // 下边这一步的结果应该是 1 2 3 4 5 6 7 8 9 10 10
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

      // 由于当前帧已经赋值给上一帧了，删除当前帧的信息
      if (USE_IMU) {
        Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
        Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
        Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

        delete pre_integrations[WINDOW_SIZE];   // 删除最后一个预积分
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }

      if (true || solver_flag == INITIAL) {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);       // 在 all_image_frame 找到边缘化的最老帧
        // 从 all_image_frame 里面删除 边缘化的最老帧
        delete it_0->second.pre_integration;    
        all_image_frame.erase(all_image_frame.begin(), it_0);
      }
      slideWindowOld();   // 处理特征观测信息
    }
  } else {    // 边缘化次新帧
    if (frame_count == WINDOW_SIZE) {

      // 将 最新帧的值赋值给最新帧，结果为 0 1 2 3 4 5 6 7 8 10 10
      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];

      if (USE_IMU) {
        // 遍历当前帧的IMU数据，将其拼接到上一帧的 IMU预积分量上
        for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
          double tmp_dt = dt_buf[frame_count][i];
          Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
          Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

          pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

          dt_buf[frame_count - 1].push_back(tmp_dt);
          linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
          angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
        }

         // 当前帧变成上一帧
        Vs[frame_count - 1] = Vs[frame_count];
        Bas[frame_count - 1] = Bas[frame_count];
        Bgs[frame_count - 1] = Bgs[frame_count];

        // 由于当前帧已经赋值给上一帧了，删除当前帧的信息
        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }
      slideWindowNew();   // 处理特征观测信息
    }
  }
}

/**
 * @brief 滑出次新帧，删除次新帧对特征点的观测
 */
void Estimator::slideWindowNew() {
  sum_of_front++;
  f_manager.removeFront(frame_count);   // 删除次新帧对特征的观测
}

/**
 * @brief 滑出最老帧，删除最老帧对特征点的观测
 */
void Estimator::slideWindowOld() {
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR ? true : false;  // 判断是否处于初始化
  if (shift_depth) {    // 初始化已经完成
    // back_R0、back_P0为窗口中最老帧的位姿
    // Rsp[0]、Ps[0] 为当前滑动窗口后第1帧的位姿，即原来的第2帧
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);   // 特征点删除最老帧的观测，将观测传递到滑窗新的第一帧（原来的第二帧）
  } else
    f_manager.removeBack();   // 特征点直接删除最老帧的观测
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[frame_count];
  T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[index];
  T.block<3, 1>(0, 3) = Ps[index];
}


/**
 * @brief 预测滑窗内的特征会在下一帧的坐标
 */
void Estimator::predictPtsInNextFrame() {
  //printf("predict pts in next frame\n");
  if (frame_count < 2)
    return;
  
  Eigen::Matrix4d curT, prevT, nextT;
  getPoseInWorldFrame(curT);                      // 获得当前帧相机的位姿 Twc
  getPoseInWorldFrame(frame_count - 1, prevT);    // 获得上一帧相机的位姿 Twc
  nextT = curT * (prevT.inverse() * curT);        // 基于匀速模型，预测下一帧的位姿
  map<int, Eigen::Vector3d> predictPts;

  // 遍历当前滑窗内的所有特征
  for (auto &it_per_id : f_manager.feature) {
    if (it_per_id.estimated_depth > 0) {      // 仅对已经初始化的路标点进行预测
      int firstIndex = it_per_id.start_frame; // 获得滑窗内第一个观测到当前特征的图像帧索引
      int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;   // 获得滑窗内最后一个观测到当前特征的图像帧索引
      //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
      if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) {   // 仅对观测次数不小于两次、且在最新图像帧中观测到的特征点进行预测
        double depth = it_per_id.estimated_depth;     // 获得在滑窗内第一个观测时的深度
        Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];  // 将特征投影到第一次观测的IMU坐标系下
        Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];                           // 特征点的世界坐标
        Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));   // 投影到下一帧IMU坐标系下
        Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);   // 投影到下一帧相机坐标系下
        int ptsIndex = it_per_id.feature_id;    // 记录特征id
        predictPts[ptsIndex] = pts_cam;         // 存储预测
      }
    }
  }
  featureTracker.setPrediction(predictPts);   // 向前端特征跟踪设置预测值
}

/**
 * @brief 计算重投影误差
 * 
 * @param[in] Ri      相机i R_wci
 * @param[in] Pi      相机i t_wci
 * @param[in] rici    相机i R_bci
 * @param[in] tici    相机i t_bci
 * @param[in] Rj      相机j R_wcj
 * @param[in] Pj      相机j t_wcj
 * @param[in] ricj    相机j R_bcj
 * @param[in] ticj    相机j t_bcj
 * @param[in] depth   特征点在相机i的深度
 * @param[in] uvi     特征点在相机i的归一化坐标
 * @param[in] uvj     特征点在相机j的归一化坐标
 * 
 * @return double     重投影误差
 */
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj) {
  Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;                     // 将特征从相机i投影特征到世界坐标系
  Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);  // 将特征从世界坐标系投影到相机j坐标系
  Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();          // 计算重投影误差
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}


/**
 * @brief 检测重投影误差，获得优化后的 Outliers
 * 
 * @param[in] removeIndex   outliers 在
 */
void Estimator::outliersRejection(set<int> &removeIndex) {
  //return;
  int feature_index = -1;
  // 遍历所有路标点
  for (auto &it_per_id : f_manager.feature) {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num = it_per_id.feature_per_frame.size();    // 获得可以观测到当前特征的滑窗内的图像帧
    if (it_per_id.used_num < 4)   // 如果当前特征观测少于4帧，则跳过
      continue;
    
    feature_index++;  // 特征计数
    int imu_i = it_per_id.start_frame;  // 获得 滑窗内观测到当前特征的第一帧
    int imu_j = imu_i - 1;              
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 获得 特征在滑窗内第一次被观测时的归一化坐标
    double depth = it_per_id.estimated_depth;               // 获得 特征在滑窗内第一次被观测时的深度

    // 遍历 滑窗内所有观测到当前特征的图像帧
    for (auto &it_per_frame : it_per_id.feature_per_frame) {  
      imu_j++;
      if (imu_i != imu_j) {   // 不同时刻，左相机在不同帧之间的重投影误差计算
        Vector3d pts_j = it_per_frame.point;
        double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                             Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                             depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
        //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
      }
      // need to rewrite projecton factor.........
      if (STEREO && it_per_frame.is_stereo) {   // 双目情形
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {   // 不同时刻，左右图像帧之间的重投影误差
          double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                               Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                               depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        } else {    // 相同时刻，左右图像帧之间的重投影误
          double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                               Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                               depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        }
      }
    }
    double ave_err = err / errCnt;    // 计算平均误差
    if (ave_err * FOCAL_LENGTH > 3)   // 若平均的重投影均方根过大，则判定该路标点为外点; 添加该路标点编号至removeIndex中
      removeIndex.insert(it_per_id.feature_id);
  }
}

/**
 * @brief IMU预积分，这里的结果作为仅IMU积分的轨迹进行发布
 * 
 * @param[in] t                     当前帧IMU的时间
 * @param[in] linear_acceleration   当前帧加速度的值
 * @param[in] angular_velocity      当前帧角速度的值
 */
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity) {
  double dt = t - latest_time;
  latest_time = t;
  Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
  Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
  latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
  latest_V = latest_V + dt * un_acc;
  latest_acc_0 = linear_acceleration;
  latest_gyr_0 = angular_velocity;
}

/**
 * @brief 
 */
void Estimator::updateLatestStates() {
  mPropagate.lock();
  latest_time = Headers[frame_count] + td;
  latest_P = Ps[frame_count];
  latest_Q = Rs[frame_count];
  latest_V = Vs[frame_count];
  latest_Ba = Bas[frame_count];
  latest_Bg = Bgs[frame_count];
  latest_acc_0 = acc_0;
  latest_gyr_0 = gyr_0;
  mBuf.lock();
  queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
  queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
  mBuf.unlock();
  while (!tmp_accBuf.empty()) {
    double t = tmp_accBuf.front().first;
    Eigen::Vector3d acc = tmp_accBuf.front().second;
    Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
    fastPredictIMU(t, acc, gyr);
    tmp_accBuf.pop();
    tmp_gyrBuf.pop();
  }
  mPropagate.unlock();
}
