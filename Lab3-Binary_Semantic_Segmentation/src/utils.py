import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    smooth = 1e-6
    pred_flat = (pred_mask > 0.5).float().view(pred_mask.size(0), -1).requires_grad_(True)  # 閾值操作，將預測轉為二元格式
    target_flat = gt_mask.view(gt_mask.size(0), -1).requires_grad_(True)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return dice

