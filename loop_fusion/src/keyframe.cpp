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

#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

/**
 * @brief 在线构建关键帧
 * 
 * @param[in] _time_stamp     时间戳
 * @param[in] _index          关键帧在当前序列中的ID
 * @param[in] _vio_T_w_i      关键帧在滑窗中的位姿 t_w1_b
 * @param[in] _vio_R_w_i      关键帧在滑窗中的位姿 R_w1_b
 * @param[in] _image          图像数据
 * @param[in] _point_3d       关键帧在滑窗中观测到的特征点 3D坐标
 * @param[in] _point_2d_uv    关键帧在滑窗中观测到的特征点 像素坐标
 * @param[in] _point_2d_norm  关键帧在滑窗中观测到的特征点 归一化坐标
 * @param[in] _point_id       关键帧在滑窗中观测到的特征点 id
 * @param[in] _sequence       关键帧所在的序列，从1开始
 */
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
                   vector<double> &_point_id, int _sequence) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  image = _image.clone();
  cv::resize(image, thumbnail, cv::Size(80, 60));
  point_3d = _point_3d;
  point_2d_uv = _point_2d_uv;
  point_2d_norm = _point_2d_norm;
  point_id = _point_id;
  has_loop = false;
  loop_index = -1;
  has_fast_point = false;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  computeWindowBRIEFPoint();  // 计算所有原有特征点的描述子
  computeBRIEFPoint();        // 对关键帧提取新的 Fast角点，并计算描述子
  if (!DEBUG_IMAGE)
    image.release();
}

/**
 * @brief 载入地图里关键帧
 * 
 * @param[in] _time_stamp     时间戳
 * @param[in] _index          关键帧在当前序列中的ID
 * @param[in] _vio_T_w_i      关键帧在滑窗中的位姿 t_w1_b   没用到
 * @param[in] _vio_R_w_i      关键帧在滑窗中的位姿 R_w1_b   没用到
 * @param[in] _T_w_i          关键帧在保存的地图中的位姿 t_w0_b
 * @param[in] _R_w_i          关键帧在保存的地图中的位姿 R_w0_b
 * @param[in] _image          图像数据
 * @param[in] _loop_index     
 * @param[in] _loop_info      
 * @param[in] _keypoints            特征点像素坐标
 * @param[in] _keypoints_norm       特征点去除畸变后的像素坐标
 * @param[in] _brief_descriptors    特征点描述子
 */
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
                   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
                   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors) {
  time_stamp = _time_stamp;
  index = _index;
  //vio_T_w_i = _vio_T_w_i;
  //vio_R_w_i = _vio_R_w_i;
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
  if (DEBUG_IMAGE) {
    image = _image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));
  }
  if (_loop_index != -1)
    has_loop = true;
  else
    has_loop = false;
  loop_index = _loop_index;
  loop_info = _loop_info;
  has_fast_point = false;
  sequence = 0;  // 保存在地图中的关键帧所在序列为0
  keypoints = _keypoints;
  keypoints_norm = _keypoints_norm;
  brief_descriptors = _brief_descriptors;
}

/**
 * @brief 计算当前关键帧在 VIO滑动窗口中 所有特征点的描述子
 */
void KeyFrame::computeWindowBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  for (int i = 0; i < (int)point_2d_uv.size(); i++) {
    cv::KeyPoint key;
    key.pt = point_2d_uv[i];
    window_keypoints.push_back(key);
  }
  extractor(image, window_keypoints, window_brief_descriptors);
}

/**
 * @brief 对当前关键帧提取新的 Fast角点，并计算描述子
 */
void KeyFrame::computeBRIEFPoint() {
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  const int fast_th = 20;  // corner detector response threshold
  if (1)
    cv::FAST(image, keypoints, fast_th, true);
  else {
    vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
    for (int i = 0; i < (int)tmp_pts.size(); i++) {
      cv::KeyPoint key;
      key.pt = tmp_pts[i];
      keypoints.push_back(key);
    }
  }
  extractor(image, keypoints, brief_descriptors);
  for (int i = 0; i < (int)keypoints.size(); i++) {
    Eigen::Vector3d tmp_p;
    m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm.push_back(tmp_norm);
  }
}

/**
 * @brief 计算描述子
 * 
 * @param[in] im						图像
 * @param[in] keys					特征点
 * @param[out] descriptors	描述子
 */
void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const {
  m_brief.compute(im, keys, descriptors);
}

/**
 * @brief 关键帧中某个特征点的描述子与回环帧的所有特征描述子匹配
 * 
 * @param[in] window_descriptor   关键帧在滑窗中观测到的某个特征点的描述子
 * @param[in] descriptors_old     回环帧的所有新提取的描述子
 * @param[in] keypoints_old       回环帧的所有新提取的特征点像素坐标
 * @param[in] keypoints_old_norm  回环帧的所有新提取特征点归一化平面坐标xy
 * @param[out] best_match         最佳匹配点像素坐标
 * @param[out] best_match_norm    最佳匹配点归一化平面坐标
 * 
 * @return true   匹配成功
 * @return false  匹配失败
 */
bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm) {
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < (int)descriptors_old.size(); i++) {
    int dis = HammingDis(window_descriptor, descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // 找到汉明距离小于80的最小值和索引即为该特征点的最佳匹配
  if (bestIndex != -1 && bestDist < 80) {
    best_match = keypoints_old[bestIndex].pt;
    best_match_norm = keypoints_old_norm[bestIndex].pt;
    return true;
  } else
    return false;
}


/**
 * @brief 匹配当前关键帧VIO中的特征 和 闭环帧候选帧新提取的特征
 * 采用匹配方式其实是暴力匹配
 * 
 * @param[out] matched_2d_old         闭环帧所有匹配特征的像素坐标
 * @param[out] matched_2d_old_norm    闭环帧所有匹配特征的归一化平面坐标xy
 * @param[out] status                 匹配状态，成功为1
 * @param[in] descriptors_old         闭环帧所有新提取特征的描述子
 * @param[in] keypoints_old           闭环帧所有新提取特征点的像素坐标
 * @param[in] keypoints_old_norm      闭环帧所有新提取特征点的归一化平面坐标xy
 */
void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm) {
  // 遍历当前特征在 VIO中提取的特征点，找到其在闭环帧中对应的匹配
  for (int i = 0; i < (int)window_brief_descriptors.size(); i++) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);

    // 将当前特征和闭环候选帧所有特征进行匹配，寻找当前特征在闭环帧中的最佳匹配
    // pt 和 pt_norm 存储最佳匹配的像素坐标和归一化坐标
    if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))   
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
    matched_2d_old_norm.push_back(pt_norm);
  }
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status) {
  int n = (int)matched_2d_cur_norm.size();
  for (int i = 0; i < n; i++)
    status.push_back(0);
  if (n >= 8) {
    vector<cv::Point2f> tmp_cur(n), tmp_old(n);
    for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++) {
      double FOCAL_LENGTH = 460.0;
      double tmp_x, tmp_y;
      tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
      tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

      tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
      tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
    }
    cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
  }
}

/**
 * @brief 利用PnP求解闭环帧在当前VIO中的位姿
 * 
 * @param[in] matched_2d_old_norm   匹配特征在闭环帧归一化平面的坐标
 * @param[in] matched_3d            匹配特征的世界坐标
 * @param[out] status               特征匹配状态，1代表成功
 * @param[out] PnP_T_old            闭环帧在当前VIO中的位姿  t_wi_old    
 * @param[out] PnP_R_old            闭环帧在当前VIO中的位姿  R_wi_old
 */
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old) {
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;

  // 当前关键帧在VIO中的 T_wc
  Matrix3d R_w_c = origin_vio_R * qic;                  // R_wc = R_wb * R_bc  
  Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;   // t_wc = R_wb * t_bc + t_wb

  // 当前关键帧的 T_bw
  R_inital = R_w_c.inverse();       // R_cw
  P_inital = -(R_inital * T_w_c);   // t_cw

  // Eigen ---> OpenCV
  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;
  TicToc t_pnp_ransac;

  // PNP算法，rvec、t 代表的是闭环帧在VIO中的 T_cw_old
  if (CV_MAJOR_VERSION < 3)
    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
  else {
    if (CV_MINOR_VERSION < 2)
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
    else
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
  }

  // 先将所有点的匹配状态标记为0
  for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
    status.push_back(0);

  // 再将内点状态标记为1
  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    status[n] = 1;
  }

  // T_cw ---> T_wc     OpenCV -----> Eigen
  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();      // R_wc_old

  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);   // t_wc_old

  // T_wb_old
  PnP_R_old = R_w_c_old * qic.transpose();
  PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

/**
 * @brief 计算当前关键帧 和 闭环帧候选帧 之间的匹配，计算两者之间的相对位姿
 * 
 * @param[in] old_kf    闭环帧
 * @return true         当前关键帧与回环帧之间有足够多的共视
 * @return false        当前关键帧与回环帧之间共视点数量少
 */
bool KeyFrame::findConnection(KeyFrame *old_kf) {
  TicToc tmp_t;
  //printf("find Connection\n");
  vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
  vector<cv::Point3f> matched_3d;
  vector<double> matched_id;
  vector<uchar> status;

  // 当前关键帧在前端VIO中观测到的特征信息
  matched_3d = point_3d;                // 世界坐标
  matched_2d_cur = point_2d_uv;         // 像素坐标
  matched_2d_cur_norm = point_2d_norm;  // 归一化坐标
  matched_id = point_id;                // ID

  TicToc t_match;
#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif

  // 利用 Brief描述子 匹配闭环帧新提取的特征 和 当前帧在VIO中观测到的特征，并剔除匹配失败的点
  // 匹配记过存储在 matched_2d_old 和 matched_2d_old_norm
  searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur_norm, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_id, status);
  //printf("search by des finish\n");

#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
#endif
  status.clear();
/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
  // 当前关键帧3d点和闭环帧2d匹配，利用PnP计算得到 闭环帧的位姿 T_bw_cur
  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  Eigen::Vector3d relative_t;
  Quaterniond relative_q;
  double relative_yaw;
  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    // 若匹配点数达到阈值，进行PnP求解
    status.clear();
    // PnP 计算闭环帧的位姿 T_w_loop
    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
    
    // 剔除误匹配
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
#if 1
    if (DEBUG_IMAGE) {
      int gap = 10;
      cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = old_kf->image;
      cv::hconcat(image, gap_image, gap_image);
      cv::hconcat(gap_image, old_img, gray_img);
      cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f cur_pt = matched_2d_cur[i];
        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_old.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (COL + gap);
        cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
      }
      cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
      putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

      putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      cv::vconcat(notation, loop_match_img, loop_match_img);

      /*
	            ostringstream path;
	            path <<  "/home/tony-ws1/raw_data/loop_image/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            */
      if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
        /*
	            	cv::imshow("loop connection",loop_match_img);  
	            	cv::waitKey(10);  
	            	*/
        cv::Mat thumbimage;
        cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
        msg->header.stamp = ros::Time(time_stamp);
        pub_match_img.publish(msg);
      }
    }
#endif
  }

  // 若PnP求解后，匹配点数达到阈值
  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);    // 计算 t_loop_cur = R_loop_w (t_w_cur - t_w_loop)
    relative_q = PnP_R_old.transpose() * origin_vio_R;                  // 计算 R_loop_cur = R_loop_w * R_w_cur
    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());   // 计算偏航角变化量

    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) { // 相对位姿 满足阈值
      has_loop = true;            // 标记找到闭环了
      loop_index = old_kf->index; // 标记 闭环帧在关键帧队列中的索引
      // 记录 闭环帧 和 当前关键帧 之间的相对位姿 T_loop_cur
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
          relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
          relative_yaw;
      //cout << "pnp relative_t " << relative_t.transpose() << endl;
      //cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
      return true;
    }
  }
  //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
  return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b) {
  BRIEF::bitset xor_of_bitset = a ^ b;
  int dis = xor_of_bitset.count();
  return dis;
}

/**
 * @brief 获得  当前关键帧在VIO中的位姿 
 * 
 * @param[out] _T_w_i   t_wb
 * @param[out] _R_w_i   R_wb
 */
void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = vio_T_w_i;
  _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = T_w_i;
  _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT() {
  return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ() {
  return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw() {
  return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info) {
  if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    //printf("update loop info\n");
    loop_info = _loop_info;
  }
}

BriefExtractor::BriefExtractor(const std::string &pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}
