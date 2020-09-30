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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};



/**
 * @brief posegraph 的关键帧数据结构
 */
class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



	double time_stamp; 
	int index;				// 关键帧在关键帧序列中的索引
	int local_index;	// 关键帧在 参与优化的关键帧序列中的索引

	Eigen::Vector3d vio_T_w_i; 			// 关键帧优化前的位姿 t_wb (会在多处地方更新) 
	Eigen::Matrix3d vio_R_w_i; 			// 关键帧优化前的位姿 R_wb (会在多处地方更新) 
	Eigen::Vector3d T_w_i;					// 关键帧优化后的位姿 t_wb
	Eigen::Matrix3d R_w_i;					// 关键帧优化后的位姿 R_wb

	Eigen::Vector3d origin_vio_T;		// 关键帧原始的（在前端VIO中的） twb
	Eigen::Matrix3d origin_vio_R;		// 关键帧原始的（在前端VIO中的） Rwb

	cv::Mat image;				// 关键帧对应的图像数据
	cv::Mat thumbnail;		
	
	// 关键帧在VIO中的数据
	vector<cv::Point3f> point_3d; 										// 特征点的世界坐标
	vector<cv::Point2f> point_2d_uv;									// 特征点的像素坐标
	vector<cv::Point2f> point_2d_norm;								// 特征点在相机归一化平面的坐标xy
	vector<double> point_id;													// 特征点的ID
	vector<cv::KeyPoint> window_keypoints;						// 关键帧在 VIO中的所有特征点（cv::KeyPoint）格式
	vector<BRIEF::bitset> window_brief_descriptors;		// 上述特征点对应的描述子

	// 关键帧在PoseGraph新提取的特征
	vector<cv::KeyPoint> keypoints;										// 新提取的特征的像素坐标
	vector<cv::KeyPoint> keypoints_norm;							// 新提取的特征的在相机归一化平面的坐标xy
	vector<BRIEF::bitset> brief_descriptors;					// 上述特征点对应的描述子

	bool has_fast_point;
	int sequence;					// 当前关键帧所在的轨迹序列

	bool has_loop;		// 当前关键帧是否存在闭环帧
	int loop_index;		// 当前关键帧的闭环帧 在关键帧序列中的索引
	Eigen::Matrix<double, 8, 1 > loop_info;		// 闭环帧 和 当前关键帧 之间的相对位姿:  T_loop_cur（7dim） + yaw_loop_cur（1dim）
};

