import os
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional, Any
import datetime

class CropClassificationDB:
    def __init__(self, host='localhost', database='crop_classification',
                 user='root', password=None, port=3306):
        if password is None:
            import os
            password = os.environ.get('MYSQL_ROOT_PASSWORD', '')
            if not password:
                import warnings
                warnings.warn('MYSQL_ROOT_PASSWORD not set — using empty password')
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            if self.connection.is_connected():
                print("✓ 数据库连接成功")
        except Error as e:
            print(f"✗ 数据库连接失败: {e}")

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ 数据库连接已关闭")

    def execute_query(self, query: str, params: tuple = None):
        cursor = None
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            return cursor.rowcount
        except Error as e:
            print(f"✗ 查询执行失败: {e}")
            self.connection.rollback()
            return -1
        finally:
            if cursor:
                cursor.close()

    def fetch_query(self, query: str, params: tuple = None) -> List[Dict]:
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except Error as e:
            print(f"✗ 查询失败: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def add_training_sample(self, opt_path: str, label_path: str, 
                           sar_path: str = None, date_acquired: str = None,
                           lon_min: float = None, lat_min: float = None,
                           lon_max: float = None, lat_max: float = None,
                           cloud_cover: float = 0, season: str = None,
                           crop_type: str = None, quality_score: int = 100):
        query = """
        INSERT INTO training_samples 
        (opt_path, sar_path, label_path, date_acquired, lon_min, lat_min, 
         lon_max, lat_max, cloud_cover, season, crop_type, quality_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE label_path = VALUES(label_path)
        """
        params = (opt_path, sar_path, label_path, date_acquired,
                  lon_min, lat_min, lon_max, lat_max,
                  cloud_cover, season, crop_type, quality_score)
        return self.execute_query(query, params)

    def add_experiment(self, exp_name: str, model_name: str = 'FusionCropNet',
                       backbone: str = 'resnet18', opt_channels: int = 10,
                       sar_channels: int = 3, num_classes: int = 7,
                       feat_dim: int = 256, lr: float = 0.001,
                       batch_size: int = 8, phase1_epochs: int = 20,
                       phase2_epochs: int = 60, pretrained: bool = True,
                       dataset_path: str = None, notes: str = None) -> int:
        query = """
        INSERT INTO experiments 
        (exp_name, model_name, backbone, opt_channels, sar_channels, 
         num_classes, feat_dim, lr, batch_size, phase1_epochs, phase2_epochs,
         pretrained, dataset_path, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (exp_name, model_name, backbone, opt_channels, sar_channels,
                  num_classes, feat_dim, lr, batch_size, phase1_epochs,
                  phase2_epochs, pretrained, dataset_path, notes)
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"✗ 添加实验失败: {e}")
            self.connection.rollback()
            return -1
        finally:
            cursor.close()

    def update_experiment_metrics(self, exp_id: int, phase: int, 
                                  miou: float, oa: float):
        if phase == 1:
            query = "UPDATE experiments SET phase1_miou = %s, phase1_oa = %s WHERE exp_id = %s"
        else:
            query = "UPDATE experiments SET phase2_miou = %s, phase2_oa = %s WHERE exp_id = %s"
        return self.execute_query(query, (miou, oa, exp_id))

    def update_experiment_best(self, exp_id: int, miou: float, oa: float):
        query = "UPDATE experiments SET best_miou = %s, best_oa = %s WHERE exp_id = %s"
        return self.execute_query(query, (miou, oa, exp_id))

    def add_time_series_scene(self, sensor_type: str, acquisition_date: str,
                              doy: int, file_path: str, tile_id: str = None,
                              band_count: int = None, resolution: int = None,
                              cloud_cover: float = 0):
        query = """
        INSERT INTO time_series_data 
        (sensor_type, acquisition_date, doy, tile_id, file_path, 
         band_count, resolution, cloud_cover)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE processed = 0
        """
        params = (sensor_type, acquisition_date, doy, tile_id, file_path,
                  band_count, resolution, cloud_cover)
        return self.execute_query(query, params)

    def add_confusion_matrix(self, exp_id: int, class_id: int, class_name: str,
                             tp: int, fp: int, fn: int):
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        
        query = """
        INSERT INTO confusion_matrix 
        (exp_id, class_id, class_name, true_positive, false_positive, 
         false_negative, precision, recall, f1_score, iou)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            true_positive = VALUES(true_positive),
            false_positive = VALUES(false_positive),
            false_negative = VALUES(false_negative),
            precision = VALUES(precision),
            recall = VALUES(recall),
            f1_score = VALUES(f1_score),
            iou = VALUES(iou)
        """
        params = (exp_id, class_id, class_name, tp, fp, fn,
                  precision, recall, f1_score, iou)
        return self.execute_query(query, params)

    def add_preprocessing_task(self, task_type: str, raw_path: str = None,
                               processed_path: str = None) -> int:
        query = """
        INSERT INTO preprocessing_tasks (task_type, raw_path, processed_path)
        VALUES (%s, %s, %s)
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (task_type, raw_path, processed_path))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"✗ 添加任务失败: {e}")
            self.connection.rollback()
            return -1
        finally:
            cursor.close()

    def update_task_status(self, task_id: int, status: str, progress: int = 0,
                           error_msg: str = None):
        query = "UPDATE preprocessing_tasks SET status = %s, progress = %s, error_msg = %s WHERE task_id = %s"
        return self.execute_query(query, (status, progress, error_msg, task_id))

    def finish_task(self, task_id: int, processed_path: str = None):
        query = """
        UPDATE preprocessing_tasks 
        SET status = 'completed', progress = 100, completed_at = NOW(), 
            processed_path = COALESCE(%s, processed_path)
        WHERE task_id = %s
        """
        return self.execute_query(query, (processed_path, task_id))

    def add_checkpoint(self, exp_id: int, epoch: int, file_path: str,
                       miou: float = None, oa: float = None):
        query = """
        INSERT INTO model_checkpoints (exp_id, epoch, file_path, miou, oa)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE miou = VALUES(miou), oa = VALUES(oa)
        """
        return self.execute_query(query, (exp_id, epoch, file_path, miou, oa))

    def add_data_quality_metrics(self, exp_id: int, avg_purity: float,
                                valid_pixel_ratio: float, cloud_coverage: float,
                                spatial_alignment_score: float,
                                missing_observation_ratio: float,
                                multi_class_boundary_ratio: float):
        query = """
        INSERT INTO data_quality_metrics 
        (exp_id, avg_purity, valid_pixel_ratio, cloud_coverage, 
         spatial_alignment_score, missing_observation_ratio, 
         multi_class_boundary_ratio)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            avg_purity = VALUES(avg_purity),
            valid_pixel_ratio = VALUES(valid_pixel_ratio),
            cloud_coverage = VALUES(cloud_coverage),
            spatial_alignment_score = VALUES(spatial_alignment_score),
            missing_observation_ratio = VALUES(missing_observation_ratio),
            multi_class_boundary_ratio = VALUES(multi_class_boundary_ratio)
        """
        params = (exp_id, avg_purity, valid_pixel_ratio, cloud_coverage,
                  spatial_alignment_score, missing_observation_ratio,
                  multi_class_boundary_ratio)
        return self.execute_query(query, params)

    def add_spatial_fold_split(self, exp_id: int, fold_idx: int,
                              train_pixel_count: int, val_pixel_count: int):
        query = """
        INSERT INTO spatial_fold_splits 
        (exp_id, fold_idx, train_pixel_count, val_pixel_count)
        VALUES (%s, %s, %s, %s)
        """
        return self.execute_query(query, (exp_id, fold_idx, train_pixel_count, val_pixel_count))

    def add_temporal_observation(self, exp_id: int, doy: int,
                                valid_count: int, total_count: int,
                                cloud_ratio: float):
        query = """
        INSERT INTO temporal_observations 
        (exp_id, doy, valid_count, total_count, cloud_ratio)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            valid_count = VALUES(valid_count),
            total_count = VALUES(total_count),
            cloud_ratio = VALUES(cloud_ratio)
        """
        return self.execute_query(query, (exp_id, doy, valid_count, total_count, cloud_ratio))

    def get_best_experiments(self, limit: int = 10) -> List[Dict]:
        query = """
        SELECT exp_id, exp_name, model_name, backbone, best_miou, best_oa, trained_at
        FROM experiments
        WHERE best_miou IS NOT NULL
        ORDER BY best_miou DESC
        LIMIT %s
        """
        return self.fetch_query(query, (limit,))

    def get_experiment_by_id(self, exp_id: int) -> Optional[Dict]:
        query = "SELECT * FROM experiments WHERE exp_id = %s"
        results = self.fetch_query(query, (exp_id,))
        return results[0] if results else None

    def get_time_series_by_sensor(self, sensor_type: str,
                                  start_date: str = None,
                                  end_date: str = None) -> List[Dict]:
        query = """
        SELECT * FROM time_series_data
        WHERE sensor_type = %s
          AND (%s IS NULL OR acquisition_date >= %s)
          AND (%s IS NULL OR acquisition_date <= %s)
        ORDER BY acquisition_date
        """
        return self.fetch_query(query, (sensor_type, start_date, start_date, end_date, end_date))

    def get_pending_tasks(self, task_type: str = None) -> List[Dict]:
        query = """
        SELECT * FROM preprocessing_tasks
        WHERE status = 'pending'
          AND (%s IS NULL OR task_type = %s)
        """
        return self.fetch_query(query, (task_type, task_type))

    def get_confusion_matrix(self, exp_id: int) -> List[Dict]:
        query = """
        SELECT class_id, class_name, true_positive, false_positive, 
               false_negative, precision, recall, f1_score, iou
        FROM confusion_matrix
        WHERE exp_id = %s
        ORDER BY class_id
        """
        return self.fetch_query(query, (exp_id,))

    def get_sample_count(self, season: str = None, crop_type: str = None) -> int:
        query = """
        SELECT COUNT(*) as count FROM training_samples
        WHERE (%s IS NULL OR season = %s)
          AND (%s IS NULL OR crop_type = %s)
        """
        results = self.fetch_query(query, (season, season, crop_type, crop_type))
        return results[0]['count'] if results else 0

    def get_data_quality_metrics(self, exp_id: int) -> Optional[Dict]:
        query = """
        SELECT avg_purity, valid_pixel_ratio, cloud_coverage,
               spatial_alignment_score, missing_observation_ratio,
               multi_class_boundary_ratio
        FROM data_quality_metrics
        WHERE exp_id = %s
        """
        results = self.fetch_query(query, (exp_id,))
        return results[0] if results else None