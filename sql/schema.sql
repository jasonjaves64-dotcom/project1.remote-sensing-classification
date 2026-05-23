CREATE DATABASE IF NOT EXISTS crop_classification DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE crop_classification;

-- 训练样本元数据表
CREATE TABLE IF NOT EXISTS training_samples (
    sample_id INT PRIMARY KEY AUTO_INCREMENT,
    opt_path VARCHAR(500) NOT NULL,
    sar_path VARCHAR(500),
    label_path VARCHAR(500) NOT NULL,
    date_acquired DATE,
    lon_min DECIMAL(10,6),
    lat_min DECIMAL(10,6),
    lon_max DECIMAL(10,6),
    lat_max DECIMAL(10,6),
    cloud_cover FLOAT DEFAULT 0,
    season VARCHAR(20),
    crop_type VARCHAR(50),
    quality_score INT DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_opt_path (opt_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 实验配置表
CREATE TABLE IF NOT EXISTS experiments (
    exp_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(50) DEFAULT 'FusionCropNet',
    backbone VARCHAR(30) DEFAULT 'resnet18',
    opt_channels INT DEFAULT 10,
    sar_channels INT DEFAULT 3,
    num_classes INT DEFAULT 7,
    feat_dim INT DEFAULT 256,
    lr FLOAT DEFAULT 0.001,
    batch_size INT DEFAULT 8,
    phase1_epochs INT DEFAULT 20,
    phase2_epochs INT DEFAULT 60,
    pretrained TINYINT(1) DEFAULT 1,
    dataset_path VARCHAR(500),
    phase1_miou FLOAT,
    phase1_oa FLOAT,
    phase2_miou FLOAT,
    phase2_oa FLOAT,
    best_miou FLOAT,
    best_oa FLOAT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    INDEX idx_model_name (model_name),
    INDEX idx_best_miou (best_miou)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 时序影像数据表
CREATE TABLE IF NOT EXISTS time_series_data (
    scene_id INT PRIMARY KEY AUTO_INCREMENT,
    sensor_type ENUM('landsat', 'sentinel1', 'sentinel2', 'modis') NOT NULL,
    acquisition_date DATE NOT NULL,
    doy INT NOT NULL,
    tile_id VARCHAR(50),
    file_path VARCHAR(500) NOT NULL,
    band_count INT,
    resolution INT,
    cloud_cover FLOAT DEFAULT 0,
    processed TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_file_path (file_path),
    INDEX idx_sensor_date (sensor_type, acquisition_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 混淆矩阵表
CREATE TABLE IF NOT EXISTS confusion_matrix (
    cm_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    class_id INT NOT NULL,
    class_name VARCHAR(30) NOT NULL,
    true_positive BIGINT DEFAULT 0,
    false_positive BIGINT DEFAULT 0,
    false_negative BIGINT DEFAULT 0,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    iou FLOAT,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_class (exp_id, class_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 预处理任务表
CREATE TABLE IF NOT EXISTS preprocessing_tasks (
    task_id INT PRIMARY KEY AUTO_INCREMENT,
    task_type ENUM('download', 'preprocess', 'patch_extract', 'fusion') NOT NULL,
    raw_path VARCHAR(500),
    processed_path VARCHAR(500),
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    progress INT DEFAULT 0,
    error_msg TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_task_type (task_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 模型检查点表
CREATE TABLE IF NOT EXISTS model_checkpoints (
    ckpt_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    epoch INT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    miou FLOAT,
    oa FLOAT,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_epoch (exp_id, epoch)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 样本分割记录表
CREATE TABLE IF NOT EXISTS data_splits (
    split_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    split_type ENUM('train', 'val', 'test') NOT NULL,
    sample_count INT DEFAULT 0,
    split_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_split (exp_id, split_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 数据质量指标表（新增）
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    dqm_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    avg_purity FLOAT,
    valid_pixel_ratio FLOAT,
    cloud_coverage FLOAT,
    spatial_alignment_score FLOAT,
    missing_observation_ratio FLOAT,
    multi_class_boundary_ratio FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_unique (exp_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 空间K折分割记录表（新增）
CREATE TABLE IF NOT EXISTS spatial_fold_splits (
    sfs_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    fold_idx INT NOT NULL,
    train_pixel_count BIGINT DEFAULT 0,
    val_pixel_count BIGINT DEFAULT 0,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_fold (exp_id, fold_idx)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 时序观测记录表（新增）
CREATE TABLE IF NOT EXISTS temporal_observations (
    to_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    doy INT NOT NULL,
    valid_count BIGINT DEFAULT 0,
    total_count BIGINT DEFAULT 0,
    cloud_ratio FLOAT DEFAULT 0,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    UNIQUE KEY idx_exp_doy (exp_id, doy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;