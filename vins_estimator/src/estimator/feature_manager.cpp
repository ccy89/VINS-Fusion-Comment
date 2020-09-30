/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

/**
 * @brief 获得当前特征在滑窗内最后一次时被哪一帧观测到 
 */
int FeaturePerId::endFrame() {
  return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs) {
  for (int i = 0; i < NUM_OF_CAM; i++)
    ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState() {
  feature.clear();
}

int FeatureManager::getFeatureCount() {
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 4) {
      cnt++;
    }
  }
  return cnt;
}

/**
 * @brief   计算每一个特征点的跟踪次数 和 它在上一帧和上上帧间的视差，判断是否是关键帧
 *          同时把新跟踪到的特征点存入 feature list 中，
 * @param   frame_count 滑动窗口内帧的个数
 * @param   image       图像观测到的所有特征点，数据结构为 <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点像素速度> > >
 * @param   td          IMU 和 cam 同步时间差
 * 
 * @return  bool    true    上一帧是关键帧; 
 *                  false   上一帧不是关键帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td) {
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;
  int parallax_num = 0;
  last_track_num = 0;         // 当前帧图像跟踪到的特征点  数目
  last_average_parallax = 0;  //
  new_feature_num = 0;        // 在当前图像 新添加的特征点 数目
  long_track_num = 0;         // 长期跟踪到的 特征点数目

  // 遍历当前帧观测到的所有特征
  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);   // 构造 当前特征点的数据结构
    assert(id_pts.second[0].first == 0);  // 判断是不是真的是左相机
    if (id_pts.second.size() == 2)  // 如果使用双目相机
    {
      f_per_fra.rightObservation(id_pts.second[1].second);  // 构造 右相机 当前特征点的数据结构
      assert(id_pts.second[1].first == 1);  // 判断是不是真的是右相机
    }

    int feature_id = id_pts.first;  // 获得 当前特征点的 id
    // 寻找 feature list 中是否有当前特征点， it 为当前特征点的 feature_id 在 feature list 中的迭代器 
    auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) {
      return it.feature_id == feature_id;
    });

    if (it == feature.end()) {  // 在 feature list 中 没有找到 当前特征点
      // 将当前特征点的 添加进 feature list
      feature.push_back(FeaturePerId(feature_id, frame_count));   // 将当前特征点的 ID
      feature.back().feature_per_frame.push_back(f_per_fra);      // 当前特征点的 绑定 ID
      new_feature_num++;  // 新特征点数目+1
    } else if (it->feature_id == feature_id) {    // 在feature list 中找到了当前特征点
      it->feature_per_frame.push_back(f_per_fra); // 存储当前图像 观测到的当前特征点的信息
      last_track_num++; 
      if (it->feature_per_frame.size() >= 4)  // 如果当前特征点被至少4帧图像观测到
        long_track_num++;
    }
  }

  // 如果当前图像为 滑动窗口起始帧2帧 | 当前图像跟踪到的特征点数目 < 20 | 长期跟踪到的特征点数目 < 40 | 新特征点数目 > 0.5 * 当前图像跟踪到的特征点数目
  // 认为上一帧图像为关键帧
  if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
    return true;

  // 遍历当前帧观测到的每一个特征点
  for (auto &it_per_id : feature) {
    // 判断上一帧和上上帧图像 是否都观测到 当前帧图像的特征点
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count); // 累加特征点的视差
      parallax_num++;   // 共同观测到的特征点数目+1
    }
  }

  if (parallax_num == 0) {  // 上上帧或上一帧没有观测到 当前帧图像的特征点
    return true;
  } else {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
    ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);

    // 如果特征点平均视差 大于 阈值，则上一帧是关键帧
    // 反之，不是关键帧
    last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;

      b = it.feature_per_frame[idx_r].point;

      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    } else
      it_per_id.solve_flag = 1;
  }
}

/**
 * @brief 删除跟踪失败的特征
 */
void FeatureManager::removeFailures() {
  // 遍历所有特征，删除跟踪失败的点
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2)    
      feature.erase(it);        // 删除特征
  }
}

void FeatureManager::clearDepth() {
  for (auto &it_per_id : feature)
    it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector() {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;
#if 1
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
    dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
  }
  return dep_vec;
}


/**
 * @brief   线性三角法 计算某个特征点的深度
 * @param   Pose0     T_c0w  相机0的投影矩阵
 * @param   Pose1     T_c1w  相机1的投影矩阵
 * @param   point0    P_c0   特征点在相机1归一化平面的坐标
 * @param   point1    P_c1   特征点在相机1归一化平面的坐标
 * @param   point_3d  P_w    特征点的世界坐标
 * 
 * @return  void
*/
void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d) {
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  
  Eigen::Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();  // 获得 P_w的齐次坐标
  
  // 齐次坐标 转 非齐次坐标
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * @brief 利用 PnP算法求解当前帧的位姿
 * 
 * @param[in] R       当前帧R的初值 R_wc
 * @param[in] P       当前帧t的初值 t_wc
 * @param[in] pts2D   特征点在当前图像归一化坐标系下的前两维
 * @param[in] pts3D   特征点的世界坐标 Pw
 * @return true       求解成功
 * @return false      求解失败
 */
bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D) {
  Eigen::Matrix3d R_initial;
  Eigen::Vector3d P_initial;

  // T_wc --> T_cw 
  R_initial = R.inverse();        // R_cw = R_wc^T
  P_initial = -(R_initial * P);   // t_cw = -R_wc^T * t_wc

  if (int(pts2D.size()) < 4) {  // 特征点数目不够
    cout << "feature tracking not enough, please slowly move you device!" << endl;
    return false;
  }

  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);    // PNP，用LM算法 求解两帧图像之间的 R t
  //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

  if (!pnp_succ) {
    printf("pnp failed ! \n");
    return false;
  }
  cv::Rodrigues(rvec, r);   // 旋转向量转为旋转矩阵

  Eigen::MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);   // R  opencv --> Eigen

  Eigen::MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);   // t  opencv --> Eigen

  // cam_T_w ---> w_T_cam
  R = R_pnp.transpose();  // R_wc = R_cw^T
  P = R * (-T_pnp);       // t_wc = -R_cw^T * t_cw   

  return true;
}


/**
 * @brief 利用PnP算法求解位姿初值
 * 
 * @param[in] frameCnt  当前帧在滑窗内的索引
 * @param[in] Ps        
 * @param[in] Rs        
 * @param[in] tic
 * @param[in] ric
 */
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]) {
  if (frameCnt > 0) {
    vector<cv::Point2f> pts2D;
    vector<cv::Point3f> pts3D;

    // 遍历所有特征点，选出已经三角化求解出深度的特征点，存入 pts3D
    for (auto &it_per_id : feature) {
      if (it_per_id.estimated_depth > 0) {
        int index = frameCnt - it_per_id.start_frame;   // 计算 当前图像在 所有观测到当前特征点的图像中的索引
        if ((int)it_per_id.feature_per_frame.size() >= index + 1) {   // 当前特征点至少在上一帧图像已经被观测到

          Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0]; // 计算 特征点在 IMU 坐标系下的坐标 P_b = R_bc * P_c + t_bc
          Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];   // 计算 特征点的世界坐标 P_w = R_wb * P_b + t_wb
          cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());  // P_w  Eigen ---> OpenCV

          // 归一化相机坐标的 x,y
          cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y());

          pts3D.push_back(point3d);
          pts2D.push_back(point2d);
        }
      }
    }
    Eigen::Matrix3d RCam;
    Eigen::Vector3d PCam;
    
    // 获得 上一帧左相机的的 T_wc  
    RCam = Rs[frameCnt - 1] * ric[0];   // R_wc = R_wb * R_bc
    PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];  // t_wc = R_wb * t_bc + t_wb

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D)) {
      // solvePoseByPnP()计算完后，RCam 和 PCam 代表 当前帧左相机的 T_wc

      // T_wc --> T_wb    T_wb = T_wc * T_cb
      Rs[frameCnt] = RCam * ric[0].transpose();   // R_wb = R_wc * R_bc^T
      Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;  // R_wb = -R_wc * R_bc^T * t_bc + t_wc

      Eigen::Quaterniond Q(Rs[frameCnt]);
      //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
      //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
    }
  }
}

/**
 * @brief 对所有特征点三角化求深度
 * 
 * @param[in] frameCnt  当前帧在滑窗内的索引
 * @param[in] Ps        滑窗内所有相机的平移量
 * @param[in] Rs        滑窗内所有相机的旋转量
 * @param[in] tic       IMU和相机之间的外参 t_bc
 * @param[in] ric       IMU和相机之间的外参 R_bc
 */
void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]) {
  // 遍历每一个特征点
  for (auto &it_per_id : feature) {
    if (it_per_id.estimated_depth > 0)  // 如果特征点深度已经计算过了，则跳过这次操作
      continue;

    if (STEREO && it_per_id.feature_per_frame[0].is_stereo) { 
      // 如果 左右相机都观测到 当前特征点

      int imu_i = it_per_id.start_frame;      // 获得 滑动窗口内 观测到当前特征点第一帧
      Eigen::Matrix<double, 3, 4> leftPose;   // 左相机的 T_cw
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];  // t_wc = R_wb * t_bc + t_wb 
      Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];    // R_wc = R_wb * R_bc
      leftPose.leftCols<3>() = R0.transpose();    // R_cw = R_wc^T
      leftPose.rightCols<1>() = -R0.transpose() * t0;   // t_cw = -R_wc^T * t_wc

      Eigen::Matrix<double, 3, 4> rightPose;    // 右相机 T_cw
      Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];  // t_wc = R_wb * t_bc + t_wb
      Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];    // R_wc = R_wb * R_bc
      rightPose.leftCols<3>() = R1.transpose();   // R_cw = R_wc^T
      rightPose.rightCols<1>() = -R1.transpose() * t1;  // t_cw = -R_wc^T * t_wc

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = it_per_id.feature_per_frame[0].point.head(2);        // 获得 当前特征点在 左相机归一化平面的坐标 x,y
      point1 = it_per_id.feature_per_frame[0].pointRight.head(2);   // 获得 当前特征点在 右相机归一化平面的坐标 x,y

      triangulatePoint(leftPose, rightPose, point0, point1, point3d); // 线性三角形法 计算当前特征点的世界坐标 P_w

      Eigen::Vector3d localPoint;   // 特征点在左相机的坐标 P_c
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();  // P_c = R_cw * P_w + t_cw
      double depth = localPoint.z();    // 获得 当前特征点在 左相机的深度
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = INIT_DEPTH;   // INIT_DEPTH = 5.0

      continue;
    } else if (it_per_id.feature_per_frame.size() > 1) {
      // 如果只有左相机观测到了 当前特征点，且当前特征点被至少两帧观测到了

      int imu_i = it_per_id.start_frame;      // 获得 滑动窗口内 观测到当前特征点第一帧(相机0)
      Eigen::Matrix<double, 3, 4> leftPose;   // 相机0的 T_cw
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];  // t_wc = R_wb * t_bc + t_wb 
      Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];              // R_wc = R_wb * R_bc
      leftPose.leftCols<3>() = R0.transpose();              // R_cw = R_wc^T
      leftPose.rightCols<1>() = -R0.transpose() * t0;       // t_cw = -R_wc^T * t_wc

      imu_i++;  // 获得 滑动窗口内 观测到当前特征点第二帧(相机1)
      Eigen::Matrix<double, 3, 4> rightPose;  // 相机1的 T_cw         
      Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];  // t_wc = R_wb * t_bc + t_wb   
      Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];              // R_wc = R_wb * R_bc
      rightPose.leftCols<3>() = R1.transpose();             // R_cw = R_wc^T
      rightPose.rightCols<1>() = -R1.transpose() * t1;      // t_cw = -R_wc^T * t_wc

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;  
      point0 = it_per_id.feature_per_frame[0].point.head(2);  // 获得 当前特征点在 相机0归一化平面的坐标 x,y
      point1 = it_per_id.feature_per_frame[1].point.head(2);  // 获得 当前特征点在 相机1归一化平面的坐标 x,y
      triangulatePoint(leftPose, rightPose, point0, point1, point3d);   // 线性三角形法 计算当前特征点的世界坐标 P_w
      Eigen::Vector3d localPoint;   // 当前特征点在相机0下的坐标
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();  // P_c = R_cw * P_w + t_cw
      double depth = localPoint.z();  // 获得 当前特征点在相机0下的深度
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = INIT_DEPTH;   // INIT_DEPTH = 5.0

      continue;
    }

    it_per_id.used_num = it_per_id.feature_per_frame.size();  // 当前特征被滑窗内几个相机观测到了
    if (it_per_id.used_num < 4)
      continue;

    /** 对特征点进行三角化求深度（SVD分解） **/
    /** 以下代码一般不会用到 **/

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;
    //R0 t0为第i帧相机坐标系到世界坐标系的变换矩阵
    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      // R t为第j帧相机坐标系到第i帧相机坐标系的变换矩阵，P为i到j的变换矩阵
      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j)
        continue;
    }
    ROS_ASSERT(svd_idx == svd_A.rows());
    // 对 A 的 SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    //it_per_id->estimated_depth = -b / A;
    //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

    it_per_id.estimated_depth = svd_method;
    //it_per_id->estimated_depth = INIT_DEPTH;

    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = INIT_DEPTH;
    }
  }
}

/**
 * @brief 丢弃特征观测队列中的outlier
 * 
 * @param[in] outlierIndex   所有outlier的ID
 */
void FeatureManager::removeOutlier(set<int> &outlierIndex) {
  std::set<int>::iterator itSet;
  // 遍历所有特征
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    int index = it->feature_id;   // 获得 特征的ID
    itSet = outlierIndex.find(index);   // 查找 当前特征是否为 outlier
    if (itSet != outlierIndex.end()) {
      feature.erase(it);    // 删除 outlier 特征
    }
  }
}

/**
 * @brief （系统已经初始化完成）边缘化最老帧时，删除最老帧的观测，传递观测量
 * 
 * @param[in] marg_R  最老帧的 Rwc
 * @param[in] marg_P  最老帧的 twc
 * @param[in] new_R   第2帧的 Rwc
 * @param[in] new_P   第2帧的 twc
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P) {
  // 遍历滑窗内的所有特征
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)   // 特征的起始观测往前移动一帧
      it->start_frame--;
    else {    
      // 如果特征的起始观测为最老帧
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;        // 获得特征在在最老帧的归一化坐标
      it->feature_per_frame.erase(it->feature_per_frame.begin());   // 从特征观测队列中删除最老帧
      if (it->feature_per_frame.size() < 2) {   
        // 如果当前特征只被最老帧 或者 最老帧+其他某一帧观测到，则删除当前特征
        feature.erase(it);
        continue;
      } else {
        // 当前特征有充足的观测
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;   // 获得当前特征在最老帧相机坐标系下的坐标
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;    // 获得特征的世界坐标
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);    // 获得 特征在第2帧相机坐标系下的坐标
        double dep_j = pts_j(2);      // 获得 特征在第2帧下的深度
        if (dep_j > 0)
          it->estimated_depth = dep_j;  // 更改当前特征的观测为第二帧的观测
        else
          it->estimated_depth = INIT_DEPTH;
      }
    }
    // remove tracking-lost feature after marginalize
    /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
  }
}


/**
 * @brief （系统未完成初始化）边缘化最老帧时，删除最老帧的观测
 */
void FeatureManager::removeBack() {
  // 遍历所有特征
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)   // 特征的起始观测往前移动一帧
      it->start_frame--;
    else {
      // 如果特征的起始观测为最老帧，直接删除对应的观测
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0)  // 如果当前特征只被最老帧观测到，删除特征
        feature.erase(it);
    }
  }
}

/**
 * @brief 边缘化次新帧的时候，删除次新帧对特征点的观测
 * 
 * @param[in] frame_count   次新帧在滑窗内的索引
 */
void FeatureManager::removeFront(int frame_count) {
  // 遍历所有特征，删除次新帧对特征的观测
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      // 如果 当前特征的第一次观测就是次新帧
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;  
      if (it->endFrame() < frame_count - 1)   // 如果次新帧没有观测到当前特征，直接跳过
        continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);   // 删除次新帧对当前特征的观测
      if (it->feature_per_frame.size() == 0)    // 如果当前特征已经没有观测了，删除当前特征
        feature.erase(it);
    }
  }
}


/**
 * @brief   计算当前特征点在上一帧图像和上上帧图像间的视差
 * @param   it_per_id   观测到 当前特征点的 所有图像合集
 * @param   frame_count 滑动窗口里图像个数
 * 
 * @return  double 视差 = srqt(du * du + dv * dv)
 *
*/
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count) {
  //check the second last frame is keyframe or not
  //parallax betwwen seconde last frame and third last frame
  const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];  // 获得 当前帧特征点 在上上次观测到的情况
  const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];  // 获得 当前帧特征点 在上次观测到的情况

  double ans = 0;
  Vector3d p_j = frame_j.point; // 获得 特征点在上次观测时的 归一化平面坐标

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point; // 获得 特征点在上上次观测时的 归一化平面坐标
  Vector3d p_i_comp;

  //int r_i = frame_count - 2;
  //int r_j = frame_count - 1;
  //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
  p_i_comp = p_i;
  // 齐次坐标转非齐次坐标     无意义，因为在定义变量的时候 p_i(2) == 1
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;  // 计算视差

  // 重复操作无意义
  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  // 等价于 ans = max(ans, sqrt(du * du + dv * dv));
  ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}