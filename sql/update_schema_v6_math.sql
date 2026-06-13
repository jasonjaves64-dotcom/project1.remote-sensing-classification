-- ============================================================================
-- V6 Mathematical Theory — Database Schema Extension
-- Adds tables for geometric invariants, topological conflict metrics,
-- TTA safety logs, and cross-module synergy tracking.
-- ============================================================================

USE crop_classification;

-- 几何不变量指标表
CREATE TABLE IF NOT EXISTS geometric_invariants (
    gi_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    sample_id VARCHAR(100),
    -- Five geometric invariants (per-image statistics)
    K_mean FLOAT COMMENT 'Mean Gaussian curvature',
    K_std FLOAT,
    H_mean FLOAT COMMENT 'Mean curvature',
    H_std FLOAT,
    k1_mean FLOAT COMMENT 'Max principal curvature',
    k1_std FLOAT,
    k2_mean FLOAT COMMENT 'Min principal curvature',
    k2_std FLOAT,
    tau_g_mean FLOAT COMMENT 'Geodesic torsion',
    tau_g_std FLOAT,
    -- SE(3) invariance verification
    K_se3_deviation FLOAT COMMENT 'Max K deviation under SE(3) transform (should be <2.3e-6)',
    -- Elevation statistics
    elev_min FLOAT,
    elev_max FLOAT,
    elev_range FLOAT,
    slope_mean FLOAT,
    slope_max FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    INDEX idx_exp_id (exp_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 拓扑冲突分析表
CREATE TABLE IF NOT EXISTS topological_conflict_metrics (
    tcm_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    epoch INT,
    phase INT COMMENT '1=training, 2=inference',
    -- Conflict classification counts
    noise_count BIGINT DEFAULT 0 COMMENT 'Pixels classified as Noise',
    structural_count BIGINT DEFAULT 0 COMMENT 'Pixels classified as Structural conflict',
    high_order_count BIGINT DEFAULT 0 COMMENT 'Pixels classified as HighOrder cyclic conflict',
    -- Aggregate metrics
    avg_kappa FLOAT COMMENT 'Mean DS conflict coefficient',
    avg_h1_norm FLOAT COMMENT 'Mean H^1 cohomology norm',
    conflict_ratio FLOAT COMMENT 'Fraction of pixels with conflict',
    -- Persistent homology metrics
    avg_persistence FLOAT COMMENT 'Mean barcode persistence length',
    n_long_lived_features INT COMMENT 'Number of long-lived topological features',
    -- Per-class conflict breakdown (JSON)
    per_class_conflict JSON COMMENT 'Class-level conflict distribution',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    INDEX idx_exp_epoch (exp_id, epoch)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- TTA安全监控日志表
CREATE TABLE IF NOT EXISTS tta_safety_logs (
    tsl_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT,
    step INT NOT NULL COMMENT 'TTA adaptation step',
    -- Three-dimensional monitor values
    gradient_alignment FLOAT COMMENT 'A_gs: geometric-semantic gradient cosine similarity',
    semantic_map FLOAT COMMENT 'Current semantic mAP estimate',
    cohomology_conflict FLOAT COMMENT 'C_coh: max H^1 norm',
    -- Intervention state
    intervention_level INT DEFAULT 0 COMMENT '0=normal, 1=light, 2=pause, 3=rollback',
    action_taken VARCHAR(50) COMMENT 'normal | halve_lr_boost_ewc | pause_tta_freeze_siren | rollback_ema_alert',
    lr_factor FLOAT COMMENT 'Learning rate multiplier applied',
    -- Recovery tracking
    auto_recovered TINYINT(1) DEFAULT 0 COMMENT 'Did system auto-recover?',
    recovery_steps INT COMMENT 'Steps needed for recovery',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_exp_step (exp_id, step)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 跨模块协同指标表
CREATE TABLE IF NOT EXISTS cross_module_synergy (
    cms_id INT PRIMARY KEY AUTO_INCREMENT,
    exp_id INT NOT NULL,
    -- Synergy 1: Geometric invariants → OT anchor quality
    geo_anchor_alignment_error FLOAT COMMENT 'Cross-region OT alignment error (should be <1e-5)',
    -- Synergy 2: Geometric invariants → conflict detection accuracy
    geo_conflict_detection_precision FLOAT,
    -- Synergy 3: T_eff → TTA safety boundary
    T_eff_estimated FLOAT COMMENT 'Effective sample size',
    T_max_safe_steps INT COMMENT 'Maximum safe TTA steps',
    -- Synergy 4: Topo-EWC → DomainAdapter adaptation
    topo_parameter_protection_ratio FLOAT COMMENT 'Fraction of parameters protected by Topo-EWC',
    adaptive_regularization_alpha FLOAT COMMENT 'Current adaptive regularization strength',
    -- Overall
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE,
    INDEX idx_exp (exp_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 实验表扩展：添加 V6 数学理论字段
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(20) DEFAULT 'V5EDL' COMMENT 'V5EDL | V5Pro | V6',
    ADD COLUMN IF NOT EXISTS siren_dem_enabled TINYINT(1) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS geometric_invariants_enabled TINYINT(1) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS topological_evidence_enabled TINYINT(1) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS grassmann_alignment_enabled TINYINT(1) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tta_enabled TINYINT(1) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS spectral_balanced_init TINYINT(1) DEFAULT 1,
    ADD COLUMN IF NOT EXISTS dem_channels INT DEFAULT 5 COMMENT '5 = standard DEM, 5 = geometric invariants';
