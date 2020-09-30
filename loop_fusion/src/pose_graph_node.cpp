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
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <visualization_msgs/Marker.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>
#include "keyframe.h"
#include "parameters.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;        // 原始图像数据
queue<sensor_msgs::PointCloudConstPtr> point_buf;   // 关键帧观测到的地图点云信息
queue<nav_msgs::Odometry::ConstPtr> pose_buf;       // 关键帧 pose
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

ros::Publisher pub_point_cloud, pub_margin_cloud;

/**
 * @brief 创建一个新的轨迹序列
 * 创建的序列号从1开始，最多为5。序列为0代表的是预先加载的地图
 */
void new_sequence() {
  printf("new sequence\n");
  sequence++;
  printf("sequence cnt %d \n", sequence);
  if (sequence > 5) {
    ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
    ROS_BREAK();
  }
  posegraph.posegraph_visualization->reset();
  posegraph.publish();
  m_buf.lock();
  while (!image_buf.empty())
    image_buf.pop();
  while (!point_buf.empty())
    point_buf.pop();
  while (!pose_buf.empty())
    pose_buf.pop();
  while (!odometry_buf.empty())
    odometry_buf.pop();
  m_buf.unlock();
}


/**
 * @brief ROS 回调函数，用于读取图像数据
 * 
 * @param[in] image_msg   图像数据
 */
void image_callback(const sensor_msgs::ImageConstPtr &image_msg) {
  //ROS_INFO("image_callback!");
  m_buf.lock();
  image_buf.push(image_msg);
  m_buf.unlock();
  //printf(" image time %f \n", image_msg->header.stamp.toSec());

  // detect unstable camera stream
  if (last_image_time == -1)
    last_image_time = image_msg->header.stamp.toSec();
  else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! detect a new sequence!");
    new_sequence();
  }
  last_image_time = image_msg->header.stamp.toSec();
}


/**
 * @brief ROS回调函数，接收VIO最新的关键帧观测到的点云数据
 * 每个点云对应一个 channel，包含信息如下:
 * [0-1] 特征点在上上帧相机的归一化坐标
 * [2-3] 特征点在上上帧相机的像素坐标
 * [4]   特征 ID
 * 
 * @param[in] point_msg   点云数据
 */
void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
  //ROS_INFO("point_callback!");
  m_buf.lock();
  point_buf.push(point_msg);    // 存储点云数据
  m_buf.unlock();
  /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i , point_msg->points[i].x, 
                                                     point_msg->points[i].y,
                                                     point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
  // for visualization
  sensor_msgs::PointCloud point_cloud;
  point_cloud.header = point_msg->header;
  for (unsigned int i = 0; i < point_msg->points.size(); i++) {
    cv::Point3f p_3d;
    p_3d.x = point_msg->points[i].x;
    p_3d.y = point_msg->points[i].y;
    p_3d.z = point_msg->points[i].z;
    Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
    geometry_msgs::Point32 p;
    p.x = tmp(0);
    p.y = tmp(1);
    p.z = tmp(2);
    point_cloud.points.push_back(p);
  }
  pub_point_cloud.publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
  sensor_msgs::PointCloud point_cloud;
  point_cloud.header = point_msg->header;
  for (unsigned int i = 0; i < point_msg->points.size(); i++) {
    cv::Point3f p_3d;
    p_3d.x = point_msg->points[i].x;
    p_3d.y = point_msg->points[i].y;
    p_3d.z = point_msg->points[i].z;
    Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
    geometry_msgs::Point32 p;
    p.x = tmp(0);
    p.y = tmp(1);
    p.z = tmp(2);
    point_cloud.points.push_back(p);
  }
  pub_margin_cloud.publish(point_cloud);
}

/**
 * @brief ROS回调函数，接收VIO最新的关键帧位姿 T_wb
 * 
 * @param[in] pose_msg  VIO最新的关键帧位姿 T_wb
 */
void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  //ROS_INFO("pose_callback!");
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
  /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n", pose_msg->pose.pose.position.x,
                                                       pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
}

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  //ROS_INFO("vio_callback!");
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
  vio_q = posegraph.w_r_vio * vio_q;

  vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
  vio_q = posegraph.r_drift * vio_q;

  nav_msgs::Odometry odometry;
  odometry.header = pose_msg->header;
  odometry.header.frame_id = "world";
  odometry.pose.pose.position.x = vio_t.x();
  odometry.pose.pose.position.y = vio_t.y();
  odometry.pose.pose.position.z = vio_t.z();
  odometry.pose.pose.orientation.x = vio_q.x();
  odometry.pose.pose.orientation.y = vio_q.y();
  odometry.pose.pose.orientation.z = vio_q.z();
  odometry.pose.pose.orientation.w = vio_q.w();
  pub_odometry_rect.publish(odometry);

  Vector3d vio_t_cam;
  Quaterniond vio_q_cam;
  vio_t_cam = vio_t + vio_q * tic;
  vio_q_cam = vio_q * qic;

  cameraposevisual.reset();
  cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
  cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
}

/**
 * @brief ROS 回调函数，接收IMU和相机之间的外参 T_bc
 * 
 * @param[in] pose_msg  IMU和相机之间的外参 T_bc
 */
void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_process.lock();
  tic = Vector3d(pose_msg->pose.pose.position.x,
                 pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                    pose_msg->pose.pose.orientation.x,
                    pose_msg->pose.pose.orientation.y,
                    pose_msg->pose.pose.orientation.z)
            .toRotationMatrix();
  m_process.unlock();
}


/**
 * @brief PoseGraph 主线程
 */
void process() {
  while (true) {
    sensor_msgs::ImageConstPtr image_msg = NULL;
    sensor_msgs::PointCloudConstPtr point_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_msg = NULL;

    // 得到具有相同时间戳的 pose_msg、image_msg、point_msg
    m_buf.lock();
    if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty()) {  
      if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec()) {   // 丢弃早于图像是位姿信息
        pose_buf.pop();
        printf("throw pose at beginning\n");
      } else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec()) {   // 丢弃早于图像的特征信息
        point_buf.pop();
        printf("throw point at beginning\n");
      } else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec()) {
        pose_msg = pose_buf.front();    // 获得最新的关键帧位姿
        pose_buf.pop();
        while (!pose_buf.empty())
          pose_buf.pop();
        while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())  // 丢弃早于关键帧位姿的图像信息
          image_buf.pop();
        image_msg = image_buf.front();  // 获得图像信息 
        image_buf.pop();

        while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())  // 丢弃遭遇关键帧位姿的特征信息
          point_buf.pop();
        point_msg = point_buf.front();  // 获得关键帧观测到的特征信息
        point_buf.pop();
      }
    }
    m_buf.unlock();

    if (pose_msg != NULL) {
      //printf(" pose time %f \n", pose_msg->header.stamp.toSec());
      //printf(" point time %f \n", point_msg->header.stamp.toSec());
      //printf(" image time %f \n", image_msg->header.stamp.toSec());

      // 不考虑最开始的几帧
      if (skip_first_cnt < SKIP_FIRST_CNT) {  
        skip_first_cnt++;
        continue;
      }

      // 每隔SKIP_CNT帧进行一次回环检测，默认 SKIP_CNT=0
      if (skip_cnt < SKIP_CNT) {  
        skip_cnt++;
        continue;
      } else {
        skip_cnt = 0;
      }

      // 将 ROS 的图像数据转换为 opencv 的数据
      cv_bridge::CvImageConstPtr ptr;
      if (image_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = image_msg->header;
        img.height = image_msg->height;
        img.width = image_msg->width;
        img.is_bigendian = image_msg->is_bigendian;
        img.step = image_msg->step;
        img.data = image_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      } else
        ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image = ptr->image;
      // 转换 当前关键帧的位姿 T_wb
      // t_wb
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                            pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);
      // R_wb
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                               pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y,
                               pose_msg->pose.pose.orientation.z)
                       .toRotationMatrix();
      
      // 运动距离大于SKIP_DIS才进行回环检测，SKIP_DIS默认为0
      if ((T - last_t).norm() > SKIP_DIS) {
        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_normal;
        vector<double> point_id;

        // 遍历当前关键帧观测到的所有特征
        for (unsigned int i = 0; i < point_msg->points.size(); i++) {
          cv::Point3f p_3d;
          // 特征点的世界坐标
          p_3d.x = point_msg->points[i].x;
          p_3d.y = point_msg->points[i].y;
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;

          // 特征点在关键帧相机归一化平面的坐标
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];

          // 特征点在关键帧的像素坐标
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];

          // 特征点ID
          p_id = point_msg->channels[i].values[4];
          
          // 存储当前特征点信息
          point_2d_normal.push_back(p_2d_normal);
          point_2d_uv.push_back(p_2d_uv);
          point_id.push_back(p_id);

          //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
        }

        // 创建关键帧
        KeyFrame *keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image,
                                          point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
        m_process.lock();
        start_flag = 1;
        posegraph.addKeyFrame(keyframe, 1);   // 向 PoseGraph 添加关键帧，开始闭环检测
        m_process.unlock();
        frame_index++;
        last_t = T;
      }
    }
    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

/**
 * @brief 键盘控制线程，用于保存posegraph 或者 创建新的序列
 */
void command() {
  while (1) {
    char c = getchar();
    if (c == 's') {
      m_process.lock();
      posegraph.savePoseGraph();  // 保存 posegraph
      m_process.unlock();
      printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
      printf("program shutting down...\n");
      ros::shutdown();
    }
    if (c == 'n')
      new_sequence();   // 创建一个新的轨迹序列

    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv) {
  // ROS初始化
  ros::init(argc, argv, "loop_fusion");
  ros::NodeHandle n("~");
  posegraph.registerPub(n);

  VISUALIZATION_SHIFT_X = 0;
  VISUALIZATION_SHIFT_Y = 0;
  SKIP_CNT = 0;
  SKIP_DIS = 0;

  if (argc != 2) {
    printf(
        "please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
        "for example: rosrun loop_fusion loop_fusion_node "
        "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
    return 0;
  }

  // 读取参数文件
  string config_file = argv[1];
  printf("config_file: %s\n", argv[1]);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  cameraposevisual.setScale(0.1);
  cameraposevisual.setLineWidth(0.01);

  std::string IMAGE_TOPIC;
  int LOAD_PREVIOUS_POSE_GRAPH;

  ROW = fsSettings["image_height"];     // 图像高度
  COL = fsSettings["image_width"];      // 图像宽度
  std::string pkg_path = ros::package::getPath("loop_fusion");
  string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
  cout << "vocabulary_file" << vocabulary_file << endl;
  posegraph.loadVocabulary(vocabulary_file);    // 加载词典文件

  // BRIEF 描述子 pattern 样式文件路径
  BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
  cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

  int pn = config_file.find_last_of('/');
  std::string configPath = config_file.substr(0, pn);
  std::string cam0Calib;
  fsSettings["cam0_calib"] >> cam0Calib;
  std::string cam0Path = configPath + "/" + cam0Calib;
  printf("cam calib path: %s\n", cam0Path.c_str());
  m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

  fsSettings["image0_topic"] >> IMAGE_TOPIC;                    // 左目图像Topic名字
  fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;   // posegraph路径
  fsSettings["output_path"] >> VINS_RESULT_PATH;                // 位姿结果输出路径 
  fsSettings["save_image"] >> DEBUG_IMAGE;                      

  LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
  VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.txt";
  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  if (!fout.good()) {
    std::cout << "Open " << VINS_RESULT_PATH << " Failed !!!!" << std::endl;
  }
  fout.close();

  int USE_IMU = fsSettings["imu"];
  // 设置 是否用到了IMU，如果使用了IMU，如果有IMU，就是4DoF优化，如果没有用到IMU，就是6DoF优化
  posegraph.setIMUFlag(USE_IMU);      
  fsSettings.release();

  // 是否加载已经存在的地图
  if (LOAD_PREVIOUS_POSE_GRAPH) {
    printf("load pose graph\n");
    m_process.lock();
    posegraph.loadPoseGraph();
    m_process.unlock();
    printf("load pose graph finish\n");
    load_flag = 1;
  } else {
    printf("no previous pose graph\n");
    load_flag = 1;
  }

  ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);
  ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);                           // 接收 图像数据
  ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 2000, pose_callback);         // 接收 VIO 上上帧（最新的关键帧）的位姿 T_wb
  ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);   // 接收 IMU与相机之间的外参
  ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);      // 接收 VIO 上上帧（最新的关键帧）观测到的所有特征
  ros::Subscriber sub_margin_point = n.subscribe("/vins_estimator/margin_cloud", 2000, margin_point_callback);

  pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
  pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
  pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud_loop_rect", 1000);
  pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud_loop_rect", 1000);
  pub_odometry_rect = n.advertise<nav_msgs::Odometry>("odometry_rect", 1000);

  std::thread measurement_process;
  std::thread keyboard_command_process;

  // pose graph主线程
  measurement_process = std::thread(process);     

  // 键盘控制线程
  keyboard_command_process = std::thread(command);

  ros::spin();

  return 0;
}
