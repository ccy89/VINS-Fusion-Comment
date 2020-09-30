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

#include "feature_tracker.h"

/**
 * @brief 判断像素点是否落在图像边缘
 * 
 * @param[in] pt  像素坐标
 * @return true   像素点不在图像边界
 * @return false  像素点在图像边界
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

// 计算 两个像素点之间的距离
double distance(cv::Point2f pt1, cv::Point2f pt2) {
  //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

// 根据 status  剔除向量中的变量
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

// 根据 status  剔除向量中的变量
void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


/**
 * @brief FeatureTracker构造函数
 */
FeatureTracker::FeatureTracker() {
  stereo_cam = 0;
  n_id = 0;
  hasPrediction = false;
}


/**
 * @brief 根据特征点被更踪到的次数挑选特征点，并保证特征点之间的像素几何距离大于30
 */
void FeatureTracker::setMask() {
  mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));   // 初始化 mask

  // 构造 特征跟踪序列，序列结构 <特征点跟踪成功次数，<特征点像素坐标，特征点id>>
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
  for (unsigned int i = 0; i < cur_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

  // 根据特征点被更踪到的次数 从大到小排列
  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
    return a.first > b.first;
  });

  // 清空变量
  cur_pts.clear();
  ids.clear();
  track_cnt.clear();

  // 挑选特征点
  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) { // 判断 mask
      cur_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      // 将 距离当前特征点 30个像素的区域标记为-1，落在该区域内的特征点将被剔除
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

// 计算 两个像素点的 像素几何距离
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2) {
  //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}


/**
 * @brief 跟踪当前帧图像的特征点
 * 
 * @param[in] _cur_time   当前帧时间戳
 * @param[in] _img        当前帧左目图像
 * @param[in] _img1       当前帧右目图像
 * @return  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
 *          特征跟踪序列，结构为 <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点速度>>>
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1) {
  TicToc t_r;
  cur_time = _cur_time;
  cur_img = _img;
  row = cur_img.rows;
  col = cur_img.cols;
  cv::Mat rightImg = _img1;
  /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
  */
  cur_pts.clear();    // 清空 当前帧特征点

  if (prev_pts.size() > 0) {  // 如果上一帧特征点不为空，即当前图像不是第一帧图像
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;
    if (hasPrediction) {  // 如果有预测值
      cur_pts = predict_pts;

      // 光流跟踪
      // 每个金字塔层的搜索窗口大小为 (21,21)
      // 图像金字塔为2层
      // 跟踪收敛条件为 迭代次数+误差大小，最大迭代次数为30，误差阈值为0.01
      // cv::OPTFLOW_USE_INITIAL_FLOW 使用预测值跟踪
      cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);  // 光流跟踪

      int succ_num = 0;
      // 统计跟踪成功的特征数目
      for (size_t i = 0; i < status.size(); i++) {  
        if (status[i])
          succ_num++;
      }
      if (succ_num < 10)
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);   // 
    } else
      cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
    
    // 反向跟踪
    if (FLOW_BACK) {
      vector<uchar> reverse_status;
      vector<cv::Point2f> reverse_pts = prev_pts;   // 
      cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
      
      // 遍历 反向跟踪的结果
      // 只保留正反跟踪都成功，且两者结果的像素几何距离 < 0.5 的点
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
          status[i] = 1;
        } else
          status[i] = 0;
      }
    }

    // 剔除 落在边界的点
    for (int i = 0; i < int(cur_pts.size()); i++) {
      if (status[i] && !inBorder(cur_pts[i]))
        status[i] = 0;
    }
    reduceVector(prev_pts, status);   // 剔除 上一帧的坏特征点
    reduceVector(cur_pts, status);    // 剔除 当前帧的坏特征点
    reduceVector(ids, status);        // 剔除 坏特征点对应的 id
    reduceVector(track_cnt, status);  // 剔除 坏特征点对应的更踪计数
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    //printf("track cnt %d\n", (int)ids.size());
  }

  // 跟踪计数+1
  for (auto &n : track_cnt)
    ++n;

  //rejectWithF();  // 通过 基本矩阵剔除outliers
  
  ROS_DEBUG("set mask begins");
  TicToc t_m;
  setMask();  // 根据特征点被更踪到的次数，特征点之间的像素几何距离大于30 挑选特征点
  ROS_DEBUG("set mask costs %fms", t_m.toc());

  ROS_DEBUG("detect feature begins");
  TicToc t_t;
  int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
  if (n_max_cnt > 0) {  // 如果特征点个数不足 MAX_CNT
    if (mask.empty())
      cout << "mask is empty " << endl;
    if (mask.type() != CV_8UC1)
      cout << "mask type wrong " << endl;
    cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);  // 提取 和当前特征不重复且像素距离大于30的 特征点
  } else
    n_pts.clear();
  ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

  // 增加 新提取的特征点
  for (auto &p : n_pts) {
    cur_pts.push_back(p);
    ids.push_back(n_id++);
    track_cnt.push_back(1);
  }
  //printf("feature cnt after add %d\n", (int)ids.size());


  cur_un_pts = undistortedPts(cur_pts, m_camera[0]);  // 获得 当前帧左目图像 矫正的特征点 归一化平面上的坐标
  pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);   // 计算当前帧左目图像 每个特征点的速度

  //  如果使用双目相机
  if (!_img1.empty() && stereo_cam) {   
    // 清空变量
    ids_right.clear();
    cur_right_pts.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();
    
    if (!cur_pts.empty()) {   // 如果左目图像提取到了特征点
      vector<cv::Point2f> reverseLeftPts;
      vector<uchar> status, statusRightLeft;
      vector<float> err;
      // cur left ---- cur right
      // 正向跟踪左右两相机的特征点
      cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
      // reverse check cur right ---- cur left
      // 反向跟踪左右两相机的特征点
      if (FLOW_BACK) {
        cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
            status[i] = 1;
          else
            status[i] = 0;
        }
      }

      ids_right = ids;
      reduceVector(cur_right_pts, status);  // 剔除 右图像坏点
      reduceVector(ids_right, status);      // 剔除 对应的 id

      cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);    // 获得 当前帧右目图像 矫正的特征点 归一化平面上的坐标
      right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);   // 计算 当前帧右目图像 每个特征点的速度
    }
    prev_un_right_pts_map = cur_un_right_pts_map;   // 当前帧变量 变 上一帧变量
  }
  if (SHOW_TRACK)
    drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

  // 更新 特征跟踪状态，当前状态变成上一帧状态
  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  prev_un_pts_map = cur_un_pts_map;
  prev_time = cur_time;
  hasPrediction = false;

  // 清空特征map
  prevLeftPtsMap.clear();   
  for (size_t i = 0; i < cur_pts.size(); i++)
    prevLeftPtsMap[ids[i]] = cur_pts[i];    // 特征根据id排序

  // 构造 特征跟踪序列，结构为 <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点速度>>>
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (size_t i = 0; i < ids.size(); i++) {
    int feature_id = ids[i];
    double x, y, z;
    x = cur_un_pts[i].x;
    y = cur_un_pts[i].y;
    z = 1;
    double p_u, p_v;
    p_u = cur_pts[i].x;
    p_v = cur_pts[i].y;
    int camera_id = 0;    // 左目
    double velocity_x, velocity_y;
    velocity_x = pts_velocity[i].x;
    velocity_y = pts_velocity[i].y;

    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  if (!_img1.empty() && stereo_cam) {
    for (size_t i = 0; i < ids_right.size(); i++) {
      int feature_id = ids_right[i];
      double x, y, z;
      x = cur_un_right_pts[i].x;
      y = cur_un_right_pts[i].y;
      z = 1;
      double p_u, p_v;
      p_u = cur_right_pts[i].x;
      p_v = cur_right_pts[i].y;
      int camera_id = 1;    // 右目
      double velocity_x, velocity_y;
      velocity_x = right_pts_velocity[i].x;
      velocity_y = right_pts_velocity[i].y;

      Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
      featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

  //printf("feature track whole time %f\n", t_r.toc());
  return featureFrame;
}

void FeatureTracker::rejectWithF() {
  if (cur_pts.size() >= 8) {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file) {
  for (size_t i = 0; i < calib_file.size(); i++) {
    ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
    camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
    m_camera.push_back(camera);
  }
  if (calib_file.size() == 2)
    stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name) {
  cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < col; i++)
    for (int j = 0; j < row; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera[0]->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
    pp.at<float>(2, 0) = 1.0;
    //cout << trackerData[0].K << endl;
    //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  // turn the following code on if you need
  // cv::imshow(name, undistortedImg);
  // cv::waitKey(0);
}

/**
 * @brief 特征点矫正畸变
 * 
 * @param[in] pts   特征点序列
 * @param[in] cam   相机模型
 * @return vector<cv::Point2f> 矫正后的特征点序列
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam) {
  vector<cv::Point2f> un_pts;
  for (unsigned int i = 0; i < pts.size(); i++) {
    Eigen::Vector2d a(pts[i].x, pts[i].y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }
  return un_pts;
}

/**
 * @brief 计算特征点的像素速度
 * 
 * @param[in] ids                 特征点id
 * @param[in] pts                 特征点归一化平面坐标
 * @param[out] cur_id_pts         当前帧 <特征点id, 特征点归一化平面坐标> 
 * @param[out] prev_id_pts        上一帧 <特征点id, 特征点归一化平面坐标> 
 * @return vector<cv::Point2f>    当前帧特征点的速度
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts) {
  // 组合当前帧特征id和归一化坐标
  vector<cv::Point2f> pts_velocity;
  cur_id_pts.clear();
  for (unsigned int i = 0; i < ids.size(); i++) {
    cur_id_pts.insert(make_pair(ids[i], pts[i]));
  }

  // 计算特征点速度
  if (!prev_id_pts.empty()) {
    double dt = cur_time - prev_time;

    for (unsigned int i = 0; i < pts.size(); i++) {
      std::map<int, cv::Point2f>::iterator it;
      it = prev_id_pts.find(ids[i]);
      if (it != prev_id_pts.end()) {
        double v_x = (pts[i].x - it->second.x) / dt;
        double v_y = (pts[i].y - it->second.y) / dt;
        pts_velocity.push_back(cv::Point2f(v_x, v_y));
      } else
        pts_velocity.push_back(cv::Point2f(0, 0));
    }
  } else {
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap) {
  //int rows = imLeft.rows;
  int cols = imLeft.cols;
  if (!imRight.empty() && stereo_cam)
    cv::hconcat(imLeft, imRight, imTrack);
  else
    imTrack = imLeft.clone();
  cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

  for (size_t j = 0; j < curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  if (!imRight.empty() && stereo_cam) {
    for (size_t i = 0; i < curRightPts.size(); i++) {
      cv::Point2f rightPt = curRightPts[i];
      rightPt.x += cols;
      cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
      //cv::Point2f leftPt = curLeftPtsTrackRight[i];
      //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  }

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < curLeftIds.size(); i++) {
    int id = curLeftIds[i];
    mapIt = prevLeftPtsMap.find(id);
    if (mapIt != prevLeftPtsMap.end()) {
      cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }
}

/**
 * @brief 设置光流跟踪的预测值
 * 
 * @param[in] predictPts    特征预测值，在相机坐标系下
 */
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts) {
  hasPrediction = true;
  predict_pts.clear();
  predict_pts_debug.clear();
  map<int, Eigen::Vector3d>::iterator itPredict;

  // 遍历前端跟踪到的所有特征
  for (size_t i = 0; i < ids.size(); i++) {
    //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
    int id = ids[i];    // 获得当前前端特征的ID
    itPredict = predictPts.find(id);  // 查找当前特征是否有预测值
    if (itPredict != predictPts.end()) {
      Eigen::Vector2d tmp_uv;
      m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);   // 将特征投影到像素平面
      predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
      predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
    } else
      predict_pts.push_back(prev_pts[i]);   // 没有预测值的，采用原始值作为预测结果
  }
}

/**
 * @brief 移除优化后为outlier的特征
 * 
 * @param[in] removePtsIds    outlier特征的ID
 */
void FeatureTracker::removeOutliers(set<int> &removePtsIds) {
  std::set<int>::iterator itSet;
  vector<uchar> status;

  // 遍历前端当前帧跟踪到的所有特征，剔除outlier
  for (size_t i = 0; i < ids.size(); i++) {
    itSet = removePtsIds.find(ids[i]);    // 获得特征ID
    if (itSet != removePtsIds.end())      // 查找当前特征是否为outlier
      status.push_back(0);    // 记录当前特征是否为outlier
    else
      status.push_back(1);
  }

  // 剔除outlier
  reduceVector(prev_pts, status);   
  reduceVector(ids, status);
  reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage() {
  return imTrack;
}