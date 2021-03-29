import torch
from torchvision.ops import box_iou


def evaluate_iou(target, pred):
    if pred["boxes"].shape[0] == 0:
        return torch.tensor(0., device=pred["boxes"].device)

    if target["boxes"].shape[0] == 0:
        return torch.tensor(0., device=target["boxes"].device)

    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


def _mean_average_precision_sub(
    pred_image_indices,
    pred_probs,
    pred_labels,
    pred_bboxes,
    target_image_indices,
    target_labels,
    target_bboxes,
    iou_threshold,
    device,
):
    classes = torch.cat([pred_labels, target_labels]).unique()
    average_precisions = torch.zeros(len(classes), device=device)

    for class_idx, c in enumerate(classes):
        desc_indices = torch.argsort(pred_probs, descending=True)
        filter = pred_labels[desc_indices] == c
        desc_indices = desc_indices[filter]

        if len(desc_indices) == 0:
            continue

        tidx = torch.unique(target_image_indices[target_labels == c])
        targets_assigned = {k.item(): {} for k in tidx}

        tps = torch.zeros(len(desc_indices), device=device)
        fps = torch.zeros(len(desc_indices), device=device)

        for i, pred_idx in enumerate(desc_indices):
            image_idx = pred_image_indices[pred_idx].item()
            gt_bboxes = target_bboxes[(
                target_image_indices == image_idx) & (target_labels == c)]
            ious = box_iou(torch.unsqueeze(
                pred_bboxes[pred_idx], dim=0), gt_bboxes)
            best_iou, best_target_idx = ious.squeeze(
                0).max(0) if len(gt_bboxes) > 0 else (0, -1)
            if best_iou > iou_threshold and not targets_assigned[image_idx].get(best_target_idx, False):
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1

        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        num_targets = len(target_labels[target_labels == c])
        recall = tps_cum / num_targets if num_targets else tps_cum
        precision = torch.cat(
            [reversed(precision), torch.tensor([1.], device=device)])
        recall = torch.cat(
            [reversed(recall), torch.tensor([0.], device=device)])
        average_precision = - \
            torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        average_precisions[class_idx] = average_precision

    mean_average_precision = torch.mean(average_precisions)

    return mean_average_precision


def mean_average_precision(preds, targets, iou_threshold, device):
    def op(x):
        return torch.cat(x).to(device)

    p_probs, p_labels, p_bboxes, p_image_indices = [], [], [], []
    for i, pred in enumerate(preds):
        p_probs.append(pred["scores"])
        p_labels.append(pred["labels"])
        p_bboxes.append(pred["boxes"])
        p_image_indices.append(torch.tensor([i] * len(pred["boxes"])))

    t_labels, t_bboxes, t_image_indices = [], [], []
    for i, target in enumerate(targets):
        t_labels.append(target["labels"])
        t_bboxes.append(target["boxes"])
        t_image_indices.append(torch.tensor([i] * len(target["boxes"])))

    p_probs, p_labels, p_bboxes, p_image_indices = list(
        map(op, [p_probs, p_labels, p_bboxes, p_image_indices]))

    t_labels, t_bboxes, t_image_indices = list(
        map(op, [t_labels, t_bboxes, t_image_indices]))

    assert len(t_labels) == len(t_bboxes) and len(
        t_labels) == len(t_image_indices)

    return _mean_average_precision_sub(p_image_indices, p_probs, p_labels, p_bboxes, t_image_indices, t_labels, t_bboxes, iou_threshold, device)


def _test_mAP(device):
    def sample3():
        def insert_dummy(preds):
            import numpy as np
            n_dummy = 1
            dummy_scores = (0.1 * np.random.rand(n_dummy)).tolist()
            dummy_labels = [99] * n_dummy
            dummy_boxes = [[1000., 1000., 1100., 1100.]] * n_dummy
            for pred in preds:
                pred["scores"] += dummy_scores
                pred["labels"] += dummy_labels
                pred["boxes"] += dummy_boxes

        box_0 = [0., 0., 100., 100.]
        box_1 = [10., 10., 110., 110.]

        preds = [
            {"scores": [0.9], "labels": [0], "boxes": [box_0]},
            {"scores": [0.9], "labels": [1], "boxes": [box_0]},
            {"scores": [0.9], "labels": [2], "boxes": [box_0]},
        ]
        targets = [
            {"labels": [0], "boxes": [box_1]},
            {"labels": [1], "boxes": [box_1]},
            {"labels": [2], "boxes": [box_1]},
        ]

        insert_dummy(preds)

        for xs in [preds, targets]:
            for x in xs:
                for k, v in x.items():
                    x[k] = torch.tensor(v, device=device)

        iou_threshold = 0.5
        args = [preds, targets, iou_threshold, device]
        expected_value = 0.75
        return args, expected_value

    def proc(args, expected):
        mAP = mean_average_precision(*args)  # mAP is torch.tensor()
        print(f"on {mAP.device} mAP:{mAP} expected mAP:{expected}")

    proc(*sample3())


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    _test_mAP(device)
