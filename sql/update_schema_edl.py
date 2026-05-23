import os
import mysql.connector
from mysql.connector import Error

def update_schema():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='crop_classification',
            user='root',
            password='',
            port=3306
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            print("更新 experiments 表 - 添加 EDL 相关字段...")
            cursor.execute("""
                ALTER TABLE experiments 
                ADD COLUMN IF NOT EXISTS uncertainty_enabled BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS edl_dropout_p FLOAT DEFAULT 0.3,
                ADD COLUMN IF NOT EXISTS edl_lambda_max FLOAT DEFAULT 0.5,
                ADD COLUMN IF NOT EXISTS edl_anneal_ep INT DEFAULT 50,
                ADD COLUMN IF NOT EXISTS avg_vacuity FLOAT,
                ADD COLUMN IF NOT EXISTS avg_dissonance FLOAT
            """)
            
            print("更新 uncertainty_metrics 表 - 添加校准字段...")
            cursor.execute("""
                ALTER TABLE uncertainty_metrics
                ADD COLUMN IF NOT EXISTS ece FLOAT,
                ADD COLUMN IF NOT EXISTS nll FLOAT,
                ADD COLUMN IF NOT EXISTS brier FLOAT
            """)

            print("创建 uncertainty_metrics 表（如不存在）...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uncertainty_metrics (
                    metric_id INT AUTO_INCREMENT PRIMARY KEY,
                    exp_id INT,
                    epoch INT,
                    phase INT,
                    vacuity_mean FLOAT,
                    vacuity_std FLOAT,
                    dissonance_mean FLOAT,
                    dissonance_std FLOAT,
                    class_var_mean FLOAT,
                    class_var_std FLOAT,
                    ece FLOAT,
                    nll FLOAT,
                    brier FLOAT,
                    FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE
                )
            """)
            
            print("更新 confusion_matrix 表 - 添加不确定性权重...")
            cursor.execute("""
                ALTER TABLE confusion_matrix 
                ADD COLUMN IF NOT EXISTS uncertainty_weight FLOAT DEFAULT 1.0
            """)
            
            connection.commit()
            print("✓ 数据库表结构更新完成")
            
    except Error as e:
        print(f"✗ 数据库更新失败: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()

if __name__ == "__main__":
    update_schema()