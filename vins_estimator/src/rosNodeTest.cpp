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

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;             // imu数据队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf;  // 特征数据队列
queue<sensor_msgs::ImageConstPtr> img0_buf;          // 左目图像数据队列
queue<sensor_msgs::ImageConstPtr> img1_buf;          // 右目图像数据队列
std::mutex m_buf;                                    // 数据获取和同步 进程锁

/**
 * @brief 获取左目图像
 * 
 * @param[in] img_msg   左目图像
 */
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img0_buf.push(img_msg);
  m_buf.unlock();
}

/**
 * @brief 获得 右目图像
 * 
 * @param[in] img_msg   右目图像
 */
void img1_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img1_buf.push(img_msg);
  m_buf.unlock();
}

/**
 * @brief ROS Image 转换为 Opencv Mat
 * 
 * @param[in] img_msg   ROS Image数据
 * @return cv::Mat Opencv 图像数据
 */
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat img = ptr->image.clone();
  return img;
}

/**
 * @brief 数据同步线程，用于获取时间戳同步的双目数据
 */
void sync_process() {
  while (1) {
    if (STEREO) {  // 双目数据
      cv::Mat image0, image1;
      std_msgs::Header header;
      double time = 0;
      m_buf.lock();
      if (!img0_buf.empty() && !img1_buf.empty()) {             // 当图像队列非空时
        double time0 = img0_buf.front()->header.stamp.toSec();  // 获得队列中最早的左目图像
        double time1 = img1_buf.front()->header.stamp.toSec();  // 获得队列中最早的右目图像
        // 判断两者的时间戳之差是否小于 3ms

        if (time0 < time1 - 0.003) {
          img0_buf.pop();
          printf("throw img0\n");
        } else if (time0 > time1 + 0.003) {
          img1_buf.pop();
          printf("throw img1\n");
        } else {
          time = img0_buf.front()->header.stamp.toSec();
          header = img0_buf.front()->header;
          image0 = getImageFromMsg(img0_buf.front());  // 从 ROS Image 获得左目图像
          img0_buf.pop();                              // 左目图像出队列
          image1 = getImageFromMsg(img1_buf.front());  // 从 ROS Image 获得右目图像
          img1_buf.pop();                              // 右目图像出队列
        }
      }
      m_buf.unlock();
      if (!image0.empty())
        estimator.inputImage(time, image0, image1);  // estimator 传入双目图像数据
    } else {                                         // 单目数据
      cv::Mat image;
      std_msgs::Header header;
      double time = 0;
      m_buf.lock();
      if (!img0_buf.empty()) {
        time = img0_buf.front()->header.stamp.toSec();
        header = img0_buf.front()->header;
        image = getImageFromMsg(img0_buf.front());  // 从 ROS Image 获得单目图像
        img0_buf.pop();
      }
      m_buf.unlock();
      if (!image.empty())
        estimator.inputImage(time, image);  // estimator 传入单目图像数据
    }

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

/**
 * @brief 接收IMU数据，并传入 estimator
 * 
 * @param[in] imu_msg   ROS IMU数据
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
  double t = imu_msg->header.stamp.toSec();
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Vector3d acc(dx, dy, dz);
  Vector3d gyr(rx, ry, rz);
  estimator.inputIMU(t, acc, gyr);
  return;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
    int feature_id = feature_msg->channels[0].values[i];
    int camera_id = feature_msg->channels[1].values[i];
    double x = feature_msg->points[i].x;
    double y = feature_msg->points[i].y;
    double z = feature_msg->points[i].z;
    double p_u = feature_msg->channels[2].values[i];
    double p_v = feature_msg->channels[3].values[i];
    double velocity_x = feature_msg->channels[4].values[i];
    double velocity_y = feature_msg->channels[5].values[i];
    if (feature_msg->channels.size() > 5) {
      double gx = feature_msg->channels[6].values[i];
      double gy = feature_msg->channels[7].values[i];
      double gz = feature_msg->channels[8].values[i];
      pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
    }
    ROS_ASSERT(z == 1);
    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }
  double t = feature_msg->header.stamp.toSec();
  estimator.inputFeature(t, featureFrame);
  return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
  if (restart_msg->data == true) {
    ROS_WARN("restart the estimator!");
    estimator.clearState();
    estimator.setParameter();
  }
  return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
  if (switch_msg->data == true) {
    //ROS_WARN("use IMU!");
    estimator.changeSensorType(1, STEREO);
  } else {
    //ROS_WARN("disable IMU!");
    estimator.changeSensorType(0, STEREO);
  }
  return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
  if (switch_msg->data == true) {
    //ROS_WARN("use stereo!");
    estimator.changeSensorType(USE_IMU, 1);
  } else {
    //ROS_WARN("use mono camera (left)!");
    estimator.changeSensorType(USE_IMU, 0);
  }
  return;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  if (argc != 2) {
    std::cout << "please intput: rosrun vins vins_node [config file] " << std::endl
              << "for example: rosrun vins vins_node ~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml" << std::endl;
    return 1;
  }

  string config_file = argv[1];
  printf("config_file: %s\n", argv[1]);

  readParameters(config_file);  // 读取参数配置
  estimator.setParameter();     // estimator 设置参数

  ROS_WARN("waiting for image and imu...");

  registerPub(n);  // 初始化 vins_estimator的publisher

  // 定义 vins_estimator 的 subscriber
  ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());  // 接收 imu 数据
  ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);             // 接收 特征数据
  ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);                                  // 接收 cam0 数据
  ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);                                  // 接收 cam1 数据
  ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);                         //
  ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);                //
  ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);                //

  std::thread sync_thread{sync_process};  // 启动数据同步线程
  ros::spin();

  return 0;
}
