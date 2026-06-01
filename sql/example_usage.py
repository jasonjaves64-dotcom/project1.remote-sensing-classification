import os
from sql.db_utils import CropClassificationDB

def main():
    db = CropClassificationDB(
        host='localhost',
        database='crop_classification',
        user=os.environ.get('MYSQL_USER', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', ''),
        port=int(os.environ.get('MYSQL_PORT', '3306'))
    )
    
    try:
        db.connect()
        
        print("\n=== 1. 添加训练样本 ===")
        sample_count = db.add_training_sample(
            opt_path='data/raw/landsat/landsat_2023_05_composite.tif',
            sar_path='data/raw/sentinel1/sentinel1_2023_05_desc.tif',
            label_path='data/raw/labels/label_2023.tif',
            date_acquired='2023-05-15',
            lon_min=115.0,
            lat_min=36.0,
            lon_max=117.0,
            lat_max=38.0,
            cloud_cover=15.5,
            season='summer',
            crop_type='冬小麦',
            quality_score=95
        )
        print(f"添加样本成功，影响行数: {sample_count}")
        
        print("\n=== 2. 创建实验记录 ===")
        exp_id = db.add_experiment(
            exp_name='FusionCropNet_v1_resnet18',
            model_name='FusionCropNet',
            backbone='resnet18',
            lr=0.001,
            batch_size=8,
            phase1_epochs=20,
            phase2_epochs=60,
            pretrained=True,
            dataset_path='data/processed/',
            notes='首次测试，使用预训练ResNet18骨干'
        )
        print(f"创建实验成功，exp_id: {exp_id}")
        
        print("\n=== 3. 更新实验指标 ===")
        db.update_experiment_metrics(exp_id, phase=1, miou=0.723, oa=0.856)
        db.update_experiment_best(exp_id, miou=0.723, oa=0.856)
        print("更新阶段1指标成功")
        
        db.update_experiment_metrics(exp_id, phase=2, miou=0.785, oa=0.892)
        db.update_experiment_best(exp_id, miou=0.785, oa=0.892)
        print("更新阶段2指标成功")
        
        print("\n=== 4. 添加混淆矩阵 ===")
        crop_classes = ['背景', '冬小麦', '夏玉米', '水稻', '大豆', '棉花', '其他']
        for i in range(1, 7):
            db.add_confusion_matrix(
                exp_id=exp_id,
                class_id=i,
                class_name=crop_classes[i],
                tp=1200 + i*100,
                fp=150 + i*20,
                fn=180 + i*25
            )
        print("添加混淆矩阵成功")
        
        print("\n=== 5. 添加时序影像 ===")
        db.add_time_series_scene(
            sensor_type='landsat',
            acquisition_date='2023-05-15',
            doy=135,
            file_path='data/raw/landsat/landsat_2023_05_composite.tif',
            tile_id='p012_r034',
            band_count=10,
            resolution=30,
            cloud_cover=15.5
        )
        print("添加时序影像成功")
        
        print("\n=== 6. 添加预处理任务 ===")
        task_id = db.add_preprocessing_task(
            task_type='preprocess',
            raw_path='data/raw/',
            processed_path='data/processed/'
        )
        print(f"创建预处理任务成功，task_id: {task_id}")
        
        db.update_task_status(task_id, 'processing', progress=50)
        print("更新任务状态为处理中")
        
        db.finish_task(task_id, processed_path='data/processed/opt_sequence.npy')
        print("任务完成")
        
        print("\n=== 7. 查询最佳实验 ===")
        best_exps = db.get_best_experiments(limit=5)
        print(f"{'exp_id':<8} {'exp_name':<30} {'backbone':<12} {'best_miou':<10} {'best_oa':<10}")
        print("-" * 70)
        for exp in best_exps:
            print(f"{exp['exp_id']:<8} {exp['exp_name']:<30} {exp['backbone']:<12} "
                  f"{exp['best_miou']:<10.4f} {exp['best_oa']:<10.4f}")
        
        print("\n=== 8. 查询样本统计 ===")
        summer_count = db.get_sample_count(season='summer')
        wheat_count = db.get_sample_count(crop_type='冬小麦')
        print(f"夏季样本数: {summer_count}")
        print(f"冬小麦样本数: {wheat_count}")
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    main()