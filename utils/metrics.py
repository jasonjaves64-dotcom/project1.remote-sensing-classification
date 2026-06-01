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
    precision_list = []
    recall_list = []
    f1_list = []
    for cls in range(1, num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().float()
        fp = ((preds == cls) & (labels != cls)).sum().float()
        fn = ((preds != cls) & (labels == cls)).sum().float()
        iou = tp / (tp + fp + fn + 1e-6)
        iou_list.append(iou.item())
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        precision_list.append(prec.item())
        recall_list.append(rec.item())
        f1_list.append((2 * prec * rec / (prec + rec + 1e-6)).item())

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
        "IoU_per_class": iou_list,
        "precision_per_class": precision_list,
        "recall_per_class": recall_list,
        "f1_per_class": f1_list,
        "confusion_matrix": cm.cpu().numpy().tolist(),
    }