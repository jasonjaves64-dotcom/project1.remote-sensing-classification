# SQL数据库架构文档

## 数据库概述

本项目使用MySQL数据库存储训练实验记录、模型性能指标和不确定性度量数据。

## 数据库表结构

### 1. experiments 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| experiment_id | INT | 主键，自增 |
| name | VARCHAR(255) | 实验名称 |
| model_version | VARCHAR(50) | 模型版本 (v4/v5/v5edl) |
| edl_enabled | BOOLEAN | 是否启用EDL不确定性估计 |
| opt_channels | INT | 光学通道数 |
| sar_channels | INT | SAR通道数 |
| dem_channels | INT | DEM通道数 |
| num_classes | INT | 类别数 |
| batch_size | INT | 批次大小 |
| epochs | INT | 训练轮数 |
| learning_rate | FLOAT | 学习率 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 2. training_logs 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| log_id | INT | 主键，自增 |
| experiment_id | INT | 外键，关联experiments |
| epoch | INT | 训练轮次 |
| train_loss | FLOAT | 训练损失 |
| val_loss | FLOAT | 验证损失 |
| train_miou | FLOAT | 训练mIoU |
| val_miou | FLOAT | 验证mIoU |
| train_oa | FLOAT | 训练OA |
| val_oa | FLOAT | 验证OA |
| timestamp | DATETIME | 记录时间 |

### 3. uncertainty_metrics 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| metric_id | INT | 主键，自增 |
| experiment_id | INT | 外键，关联experiments |
| epoch | INT | 训练轮次 |
| avg_vacuity | FLOAT | 平均数据不确定性 |
| avg_dissonance | FLOAT | 平均认知不确定性 |
| avg_class_variance | FLOAT | 平均类别方差 |
| timestamp | DATETIME | 记录时间 |

### 4. confusion_matrix 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| cm_id | INT | 主键，自增 |
| experiment_id | INT | 外键，关联experiments |
| epoch | INT | 训练轮次 |
| matrix_json | TEXT | 混淆矩阵（JSON格式） |
| uncertainty_weight | FLOAT | 不确定性权重 |
| timestamp | DATETIME | 记录时间 |

### 5. model_checkpoints 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| checkpoint_id | INT | 主键，自增 |
| experiment_id | INT | 外键，关联experiments |
| epoch | INT | 保存轮次 |
| path | VARCHAR(500) | 模型路径 |
| metric_value | FLOAT | 验证指标值 |
| is_best | BOOLEAN | 是否最佳模型 |
| saved_at | DATETIME | 保存时间 |

### 6. data_records 表

| 字段名 | 类型 | 说明 |
|--------|------|------|
| record_id | INT | 主键，自增 |
| experiment_id | INT | 外键，关联experiments |
| data_type | VARCHAR(50) | 数据类型 (optical/sar/dem/label) |
| path | VARCHAR(500) | 数据路径 |
| sample_count | INT | 样本数量 |
| created_at | DATETIME | 创建时间 |

## 表关系图

```
experiments
    │
    ├── 1:N training_logs
    │
    ├── 1:N uncertainty_metrics
    │
    ├── 1:N confusion_matrix
    │
    ├── 1:N model_checkpoints
    │
    └── 1:N data_records
```

## SQL初始化脚本

```sql
-- 创建数据库
CREATE DATABASE IF NOT EXISTS crop_classification;

-- 使用数据库
USE crop_classification;

-- 创建 experiments 表
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) DEFAULT 'v5',
    edl_enabled BOOLEAN DEFAULT FALSE,
    opt_channels INT DEFAULT 10,
    sar_channels INT DEFAULT 5,
    dem_channels INT DEFAULT 5,
    num_classes INT DEFAULT 7,
    batch_size INT DEFAULT 8,
    epochs INT DEFAULT 80,
    learning_rate FLOAT DEFAULT 0.001,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 创建 training_logs 表
CREATE TABLE IF NOT EXISTS training_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    epoch INT NOT NULL,
    train_loss FLOAT,
    val_loss FLOAT,
    train_miou FLOAT,
    val_miou FLOAT,
    train_oa FLOAT,
    val_oa FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- 创建 uncertainty_metrics 表
CREATE TABLE IF NOT EXISTS uncertainty_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    epoch INT NOT NULL,
    avg_vacuity FLOAT,
    avg_dissonance FLOAT,
    avg_class_variance FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- 创建 confusion_matrix 表
CREATE TABLE IF NOT EXISTS confusion_matrix (
    cm_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    epoch INT NOT NULL,
    matrix_json TEXT,
    uncertainty_weight FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- 创建 model_checkpoints 表
CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    epoch INT NOT NULL,
    path VARCHAR(500) NOT NULL,
    metric_value FLOAT,
    is_best BOOLEAN DEFAULT FALSE,
    saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- 创建 data_records 表
CREATE TABLE IF NOT EXISTS data_records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    path VARCHAR(500) NOT NULL,
    sample_count INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

## 使用示例

```sql
-- 插入实验记录
INSERT INTO experiments (name, model_version, edl_enabled, batch_size)
VALUES ('EDL_Experiment_001', 'v5edl', TRUE, 8);

-- 查询实验的不确定性指标
SELECT epoch, avg_vacuity, avg_dissonance 
FROM uncertainty_metrics 
WHERE experiment_id = 1 
ORDER BY epoch;
```