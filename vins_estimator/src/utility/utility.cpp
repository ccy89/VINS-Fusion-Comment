/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

/**
 * @brief // 计算当前加速度计值和重力向量之间的旋转 R_wb
 * 
 * @param[in] g     当前加速度计的值
 * @return Eigen::Matrix3d  
 */
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();   // 计算当前加速度计值和重力向量之间的旋转 R_wb
    double yaw = Utility::R2ypr(R0).x();    // 获得 偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 设置偏航角为0，由于没有磁力计，偏航角不可观
    return R0;
}
