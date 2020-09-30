/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"

void ResidualBlockInfo::Evaluate() {
  residuals.resize(cost_function->num_residuals());   // 初始化残差维度

  std::vector<int> block_sizes = cost_function->parameter_block_sizes();  // 获得每个变量的维度
  raw_jacobians = new double *[block_sizes.size()];
  jacobians.resize(block_sizes.size());   // 残差对每个变量的 Jacobian

  // 遍历所有变量
  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
    jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);  // 初始化 残差对当前变量的 Jacobian矩阵维度 
    raw_jacobians[i] = jacobians[i].data();                               // 映射当前Jacobian矩阵的地址
    //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
  }
  cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);  // 计算 残差和Jacobian

  //std::vector<int> tmp_idx(block_sizes.size());
  //Eigen::MatrixXd tmp(dim, dim);
  //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
  //{
  //    int size_i = localSize(block_sizes[i]);
  //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
  //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
  //    {
  //        int size_j = localSize(block_sizes[j]);
  //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
  //        tmp_idx[j] = sub_idx;
  //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
  //    }
  //}
  //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
  //std::cout << saes.eigenvalues() << std::endl;
  //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

  // 计算 使用鲁棒核函数修正Jacobian 和 残差
  if (loss_function) {
    double residual_scaling_, alpha_sq_norm_;

    double sq_norm, rho[3];

    sq_norm = residuals.squaredNorm();
    loss_function->Evaluate(sq_norm, rho);
    //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

    double sqrt_rho1_ = sqrt(rho[1]);

    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    } else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }

    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
    }

    residuals *= residual_scaling_;
  }
}

MarginalizationInfo::~MarginalizationInfo() {
  //ROS_WARN("release marginlizationinfo");

  for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
    delete it->second;

  for (int i = 0; i < (int)factors.size(); i++) {
    delete[] factors[i]->raw_jacobians;

    delete factors[i]->cost_function;

    delete factors[i];
  }
}

/**
 * @brief 添加边缘化残差块
 * 
 * @param[in] residual_block_info  边缘化残差块
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info) {
  factors.emplace_back(residual_block_info);  // 存储残差块

  std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;    // 获得残差块关联的每个变量在内存的起始地址
  // 获得每个变量的维度
  // IMU残差是 7，9，7，9
  // 视觉残差是 7，7，7，1 或者 7，7，7，1，1
  std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

  // 遍历残差块所有变量，对于 IMU残差是4个 ，对于视觉残差是4个或5个（考虑 td）
  for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
    double *addr = parameter_blocks[i];   // 获得当前变量在内存中的地址
    int size = parameter_block_sizes[i];  // 获得当前变量的 global维度
    parameter_block_size[reinterpret_cast<long>(addr)] = size;  // 组合 当前变量的地址 和 当前变量的 global维度
  }

  // 遍历当前残差块需要边缘化的变量，初始化其在Hessian矩阵中的所用为0
  for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++) {
    double *addr = parameter_blocks[residual_block_info->drop_set[i]];  // 获得边缘化变量在内存中的地址
    parameter_block_idx[reinterpret_cast<long>(addr)] = 0;              // 边缘化变量在 Hessian矩阵中的索引，初始化为0          
  }
}

/**
 * @brief 边缘化预处理
 *        遍历所有残差块，计算Jacobian矩阵 和 残差(residuals)
 *        将变量数据放到 parameter_block_data之中
 */
void MarginalizationInfo::preMarginalize() {
  for (auto it : factors) {
    it->Evaluate();   // 计算Jacobian矩阵 和 残差(residuals)

    std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();  // 获得当前残差块每个变量的 global维度

    // 遍历当前残差块的每个变量，将变量数据放到 parameter_block_data之中
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
      long addr = reinterpret_cast<long>(it->parameter_blocks[i]);    // 获得当前变量的内存地址
      int size = block_sizes[i];                                      // 获得当前变量的global维度
      if (parameter_block_data.find(addr) == parameter_block_data.end()) {  // 如果当前变量还没有添加过
        double *data = new double[size];
        memcpy(data, it->parameter_blocks[i], sizeof(double) * size); // 拷贝变量数据
        parameter_block_data[addr] = data;  // 组合 变量地址 和 变量数据
      }
    }
  }
}

int MarginalizationInfo::localSize(int size) const {
  return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const {
  return size == 6 ? 7 : size;
}

/**
 * @brief 单线程构建 Hessian矩阵
 * 
 * @param[in] threadsstruct   当前线程
 */
void *ThreadsConstructA(void *threadsstruct) {
  ThreadsStruct *p = ((ThreadsStruct *)threadsstruct);

  // 遍历当前线程负责的所有残差块，计算对应的Hessian矩阵块
  for (auto it : p->sub_factors) {
    // 遍历当前残差块的所有变量
    for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
      int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];    // 获得变量i 在Hessian中的索引 
      int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];  // 获得变量i 在Hessian中的global维度
      if (size_i == 7)    // 转换 global维度 到 local维度 （只有相机位姿的global维度和local维度不一样）
        size_i = 6;
      Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);   // 获得残差对变量i的Jacobian

      for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
        int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];    // 获得变量j 在Hessian中的索引 
        int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];  // 获得变量j 在Hessian中的global维度
        if (size_j == 7)  // 转换 global维度 到 local维度 （只有相机位姿的global维度和local维度不一样）
          size_j = 6;
        Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);   // 获得残差对变量j的Jacobian
        if (i == j)
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;  // 对角线
        else {
          // 非对角线，存在对称的一个矩阵块
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;  
          p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
        }
      }
      p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;  // 计算 b
    }
  }
  return threadsstruct;
}

/**
 * @brief 执行边缘化操作
 *        构建 Ax=b，计算舒尔补
 */
void MarginalizationInfo::marginalize() {
  int pos = 0;    // Hessian矩阵的维度

  // 先遍历边缘化的变量，把他们放在Hessian的前边
  for (auto &it : parameter_block_idx) {
    it.second = pos;    // 当前变量在 Hessian 中的索引
    pos += localSize(parameter_block_size[it.first]);
  }
  m = pos;  // 待边缘化变量的维度

  // 遍历非边缘化的变量，存储其在Hessian中的索引
  for (const auto &it : parameter_block_size) {
    if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
      parameter_block_idx[it.first] = pos;  // 当前变量在 Hessian 中的索引
      pos += localSize(it.second);
    }
  }

  n = pos - m;  // 非边缘化变量的维度
  //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
  if (m == 0) {
    valid = false;
    printf("unstable tracking...\n");
    return;
  }

  
  TicToc t_summing;
  Eigen::MatrixXd A(pos, pos);    // 设置 Hessian矩阵维度
  Eigen::VectorXd b(pos);         // 设置 b矩阵维度
  A.setZero();
  b.setZero();

  // 多线程
  TicToc t_thread_summing;
  pthread_t tids[NUM_THREADS];
  ThreadsStruct threadsstruct[NUM_THREADS];
  int i = 0;

  // 遍历所有残差块，将其分成 NUM_THREADS 份
  for (auto it : factors) {
    threadsstruct[i].sub_factors.push_back(it);
    i++;
    i = i % NUM_THREADS;
  }

  // 多线程构建 Hessian 矩阵
  for (int i = 0; i < NUM_THREADS; i++) {
    TicToc zero_matrix;
    threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
    threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
    threadsstruct[i].parameter_block_size = parameter_block_size;
    threadsstruct[i].parameter_block_idx = parameter_block_idx;
    int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *)&(threadsstruct[i])); // 创建线程
    if (ret != 0) {
      ROS_WARN("pthread_create error");
      ROS_BREAK();
    }
  }

  // 将所有线程的结果相加
  for (int i = NUM_THREADS - 1; i >= 0; i--) {
    pthread_join(tids[i], NULL);
    A += threadsstruct[i].A;
    b += threadsstruct[i].b;
  }
  //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
  //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

  /********************** 边缘化操作 *******************/ 

  // Hessian 矩阵是对称矩阵，该表达式等价于 Amm = A.block(0, 0, m, m)
  Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());

  // 对 Hessian 矩阵进行特征分解，获得 特征向量 和特征值
  // 一个对称矩阵一定可以分解成  U Σ U^T    U是特征向量（而且是正交矩阵），Σ是特征值组成的对角矩阵
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

  // 求解Amm的逆矩阵，矩阵和矩阵的逆，特征值取倒数，特征向量一样。
  // Eigen select 函数：  (矩阵A条件语句).select(表达式1，表达式2) 
  // 遍历矩阵元素，如果矩阵元素 A(i,j) 满足条件语句，则 A(i,j) = 表达式1，否则 A(i,j) = 表达式2
  // 这里将 特征值中较小的值 直接设为0
  Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

  Eigen::VectorXd bmm = b.segment(0, m);
  Eigen::MatrixXd Amr = A.block(0, m, m, n);
  Eigen::MatrixXd Arm = A.block(m, 0, n, m);
  Eigen::MatrixXd Arr = A.block(m, m, n, n);
  Eigen::VectorXd brr = b.segment(m, n);

  // 舒尔补
  A = Arr - Arm * Amm_inv * Amr;    // 依然是一个对称矩阵
  b = brr - Arm * Amm_inv * bmm;

  // 将先验信息进行拆解，以满足ceres内部求解原理，A = J^T * J  
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));                // 获得 A 的特征向量 
  Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));  // 获得 A^{-1} 的特征向量 

  // 特征值开根号
  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();  // J = √Σ * U^T ======>  J^T * J = U * √Σ * √Σ * U^T = U * Σ * U^T = A
  linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;  // res = 1 / √Σ * U^T * b  ====> J^T * res = U^T * √Σ  * (1 / √Σ) * U^T * b = b
}

/**
 * @brief 获得当前所有非边缘化变量的地址
 * 
 * @param[in] addr_shift          当前边缘化 除特征点外的所有变量的地址
 * @return std::vector<double *>  非边缘化变量的地址
 */ 
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift) {
  std::vector<double *> keep_block_addr;
  keep_block_size.clear();
  keep_block_idx.clear();
  keep_block_data.clear();

  // 遍历所有变量 
  for (const auto &it : parameter_block_idx) {
    if (it.second >= m) {   
      // 只考虑非边缘化的变量
      keep_block_size.push_back(parameter_block_size[it.first]);  // 变量维度
      keep_block_idx.push_back(parameter_block_idx[it.first]);    // 变脸在Hessian中的索引
      keep_block_data.push_back(parameter_block_data[it.first]);  // 变量数据
      keep_block_addr.push_back(addr_shift[it.first]);            // 变量新地址
    }
  }
  sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

  return keep_block_addr;
}

/**
 * @brief 构造边缘化因子
 * 
 * @param[in] _marginalization_info   边缘化信息体
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info(_marginalization_info) {
  int cnt = 0;
  for (auto it : marginalization_info->keep_block_size) { // 遍历先验信息所有变量的维度
    mutable_parameter_block_sizes()->push_back(it);       // 存储维度
    cnt += it;  // 计算总维度
  }
  //printf("residual size: %d, %d\n", cnt, n);
  set_num_residuals(marginalization_info->n);   // 设置残差维度
};

/**
 * @brief 先验信息计算Jacobian 和 残差，重载 ceres::CostFunction::Evaluate
 * 
 * @param[in] parameters    CostFunction 相关的所有变量
 * @param[in] residuals     CostFunction 的残差
 * @param[in] jacobians     残差对所有变量的Jacobian矩阵
 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  int n = marginalization_info->n;    // 非边缘化变量的维度
  int m = marginalization_info->m;    // 边缘化变量的总维度
  Eigen::VectorXd dx(n);    // 残差

  // 遍历先验信息
  for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
    int size = marginalization_info->keep_block_size[i];    // 获得当前变量的 global维度
    int idx = marginalization_info->keep_block_idx[i] - m;  // 获得当前变量在先验信息矩阵中的起始索引
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);   // 获得变量当前的值
    Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);   // 获得变量在先验矩阵中的线性化点
    
    // 计算 Δx
    if (size != 7)
      dx.segment(idx, size) = x - x0;   // 速度和偏置的Δx
    else {
      // 旋转平移 Δx
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0)) {
        dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      }
    }
  }

  // 更新残差，不然优化会崩溃，b = b + A * dx
  Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

  // 下面这段代码是 jacobian = marginalization_info->linearized_jacobians
  // 意味 Jacobian矩阵不变 === 先验信息在优化过程中不变
  if (jacobians) {
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) {
      if (jacobians[i]) {
        int size = marginalization_info->keep_block_size[i];      // 变量的 global维度
        int local_size = marginalization_info->localSize(size);   // 变量的 local维度
        int idx = marginalization_info->keep_block_idx[i] - m;    // 变量在先验信息矩阵中的起始索引
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
        jacobian.setZero();
        // 从 linearized_jacobians 中提取出残差对当前变量的 Jacobian 矩阵
        jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
      }
    }
  }
  return true;
}
