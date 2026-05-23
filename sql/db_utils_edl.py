import os
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional, Any
import datetime

class CropClassificationDBEDL:
    def __init__(self, host='localhost', database='crop_classification', 
                 user='root', password='', port=3306):
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

    def add_experiment_edl(self, exp_name: str, model_name: str = 'FusionCropNetV5EDL',
                           backbone: str = 'resnet50', opt_channels: int = 10,
                           sar_channels: int = 5, dem_channels: int = 5,
                           num_classes: int = 7, feat_dim: int = 512, lr: float = 0.001,
                           batch_size: int = 8, phase1_epochs: int = 20,
                           phase2_epochs: int = 60, pretrained: bool = True,
                           dataset_path: str = None, notes: str = None,
                           uncertainty_enabled: bool = True,
                           edl_dropout_p: float = 0.3,
                           edl_lambda_max: float = 0.5,
                           edl_anneal_ep: int = 50) -> int:
        query = """
        INSERT INTO experiments 
        (exp_name, model_name, backbone, opt_channels, sar_channels, 
         num_classes, feat_dim, lr, batch_size, phase1_epochs, phase2_epochs,
         pretrained, dataset_path, notes, uncertainty_enabled,
         edl_dropout_p, edl_lambda_max, edl_anneal_ep)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (exp_name, model_name, backbone, opt_channels, sar_channels,
                  num_classes, feat_dim, lr, batch_size, phase1_epochs,
                  phase2_epochs, pretrained, dataset_path, notes,
                  uncertainty_enabled, edl_dropout_p, edl_lambda_max, edl_anneal_ep)
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"✗ 添加EDL实验失败: {e}")
            self.connection.rollback()
            return -1
        finally:
            cursor.close()

    def update_experiment_uncertainty(self, exp_id: int, vacuity: float, dissonance: float):
        query = """
        UPDATE experiments 
        SET avg_vacuity = %s, avg_dissonance = %s 
        WHERE exp_id = %s
        """
        return self.execute_query(query, (vacuity, dissonance, exp_id))

    def add_uncertainty_metrics(self, exp_id: int, epoch: int, phase: int,
                                vacuity_mean: float, vacuity_std: float,
                                dissonance_mean: float, dissonance_std: float,
                                class_var_mean: float = None, class_var_std: float = None,
                                ece: float = None, nll: float = None, brier: float = None):
        query = """
        INSERT INTO uncertainty_metrics
        (exp_id, epoch, phase, vacuity_mean, vacuity_std,
         dissonance_mean, dissonance_std, class_var_mean, class_var_std,
         ece, nll, brier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            vacuity_mean = VALUES(vacuity_mean),
            vacuity_std = VALUES(vacuity_std),
            dissonance_mean = VALUES(dissonance_mean),
            dissonance_std = VALUES(dissonance_std),
            class_var_mean = VALUES(class_var_mean),
            class_var_std = VALUES(class_var_std),
            ece = VALUES(ece),
            nll = VALUES(nll),
            brier = VALUES(brier)
        """
        params = (exp_id, epoch, phase, vacuity_mean, vacuity_std,
                  dissonance_mean, dissonance_std, class_var_mean, class_var_std,
                  ece, nll, brier)
        return self.execute_query(query, params)

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

    def add_confusion_matrix(self, exp_id: int, class_id: int, class_name: str,
                             tp: int, fp: int, fn: int, uncertainty_weight: float = 1.0):
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        
        query = """
        INSERT INTO confusion_matrix 
        (exp_id, class_id, class_name, true_positive, false_positive, 
         false_negative, precision, recall, f1_score, iou, uncertainty_weight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            true_positive = VALUES(true_positive),
            false_positive = VALUES(false_positive),
            false_negative = VALUES(false_negative),
            precision = VALUES(precision),
            recall = VALUES(recall),
            f1_score = VALUES(f1_score),
            iou = VALUES(iou),
            uncertainty_weight = VALUES(uncertainty_weight)
        """
        params = (exp_id, class_id, class_name, tp, fp, fn,
                  precision, recall, f1_score, iou, uncertainty_weight)
        return self.execute_query(query, params)

    def add_checkpoint(self, exp_id: int, epoch: int, file_path: str,
                       miou: float = None, oa: float = None):
        query = """
        INSERT INTO model_checkpoints (exp_id, epoch, file_path, miou, oa)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE miou = VALUES(miou), oa = VALUES(oa)
        """
        return self.execute_query(query, (exp_id, epoch, file_path, miou, oa))

    def get_best_experiments_with_uncertainty(self, limit: int = 10) -> List[Dict]:
        query = """
        SELECT exp_id, exp_name, model_name, backbone, best_miou, best_oa, 
               uncertainty_enabled, avg_vacuity, avg_dissonance, trained_at
        FROM experiments
        WHERE best_miou IS NOT NULL
        ORDER BY best_miou DESC
        LIMIT %s
        """
        return self.fetch_query(query, (limit,))

    def get_uncertainty_metrics(self, exp_id: int) -> List[Dict]:
        query = """
        SELECT epoch, phase, vacuity_mean, vacuity_std, 
               dissonance_mean, dissonance_std, class_var_mean, class_var_std
        FROM uncertainty_metrics
        WHERE exp_id = %s
        ORDER BY epoch
        """
        return self.fetch_query(query, (exp_id,))