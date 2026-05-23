import torch

def compute_metrics(preds: torch.Tensor,
                    labels: torch.Tensor,
                    num_classes: int) -> dict:
    """
    计算 OA、Kappa、每类IoU 和 mIoU
    """
    valid_mask = labels != 255
    preds      = preds[valid_mask]
    labels     = labels[valid_mask]

    oa = (preds == labels).float().mean().item()

    iou_list = []
    for cls in range(1, num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().float()
        fp = ((preds == cls) & (labels != cls)).sum().float()
        fn = ((preds != cls) & (labels == cls)).sum().float()
        iou = tp / (tp + fp + fn + 1e-6)
        iou_list.append(iou.item())

    miou = sum(iou_list) / len(iou_list)

    n  = len(preds)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t.long(), p.long()] += 1

    po = cm.diag().sum().float() / n
    pe = (cm.sum(0).float() * cm.sum(1).float()).sum() / (n ** 2)
    kappa = ((po - pe) / (1 - pe + 1e-6)).item()

    return {
        "OA"       : oa,
        "mIoU"     : miou,
        "Kappa"    : kappa,
        "IoU_per_class": iou_list
    }