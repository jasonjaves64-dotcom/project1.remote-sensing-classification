import torch
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os

def objective(config):
    from models.fusion_net_v5 import FusionCropNetV5
    from models.fusion_net_v5pro import FusionCropNetV5Pro

    if config.get("model_type", "").lower() == "v5pro":
        model = FusionCropNetV5Pro(
            opt_ch=10,
            sar_ch=5,
            dem_ch_in=5,
            num_classes=7,
            feat_dim=config.get("feat_dim", 512),
            backbone=config.get("backbone", "resnet50"),
            pretrained=False,
            n_heads=config.get("n_heads", 16),
            win_size=4,
            n_layers=config.get("n_layers", 4)
        )
    else:
        model = FusionCropNetV5(
            opt_ch=10,
            sar_ch=5,
            dem_ch_in=5,
            num_classes=7,
            feat_dim=config.get("feat_dim", 512),
            backbone=config.get("backbone", "resnet50"),
            pretrained=False,
            n_heads=config.get("n_heads", 16),
            win_size=4,
            n_layers=config.get("n_layers", 4)
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainer = TwoPhaseTrainer(model, device)
    
    try:
        if train_loader is None or val_loader is None:
            print("Warning: HPO requires real data loaders. "
                  "Set up train_loader/val_loader before calling train_model().")
            return {"val_miou": 0}

        history_p1 = trainer.train_phase1(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get("epochs_p1", 10),
            lr=config.get("lr_p1", 0.001),
        )

        history_p2 = trainer.train_phase2(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get("epochs_p2", 30),
            lr=config.get("lr_p2", 0.0001),
        )

        val_miou = history_p2["val_miou"][-1] if history_p2 else 0

        return {"val_miou": val_miou}

    except Exception as e:
        print(f"Training failed: {e}")
        return {"val_miou": 0}

def run_hpo(config_space=None, num_samples=20, max_epochs=40):
    if config_space is None:
        config_space = {
            "feat_dim": tune.choice([32, 64, 128]),
            "backbone": tune.choice(["resnet18", "resnet34"]),
            "epochs_p1": tune.randint(5, 20),
            "epochs_p2": tune.randint(20, 60),
            "lr_p1": tune.loguniform(1e-4, 1e-2),
            "lr_p2": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([4, 8, 16]),
            "weight_decay": tune.loguniform(1e-6, 1e-4)
        }
    
    scheduler = ASHAScheduler(
        metric="val_miou",
        mode="max",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["val_miou", "training_iteration"]
    )
    
    analysis = tune.run(
        objective,
        config=config_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        local_dir="ray_results"
    )
    
    print("✅ 超参数优化完成")
    print(f"最佳验证mIoU: {analysis.best_result['val_miou']}")
    print(f"最佳配置: {analysis.best_config}")
    
    return analysis

def save_best_config(analysis, output_path="best_config.yaml"):
    import yaml
    
    best_config = analysis.best_config
    best_config["val_miou"] = analysis.best_result["val_miou"]
    
    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"✅ 最佳配置已保存: {output_path}")