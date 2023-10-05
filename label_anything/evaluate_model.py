import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        # penso che tutti questi assegnamenti possano essere eliminati
        # utilizzando qualcosa come **batch_dict quando si passano
        # i dati al modello
        target, examples, p_bboxes, p_masks, p_points, gt = (
            batch_dict["target"].to(device),
            batch_dict["example"].to(device),
            batch_dict["p_bbox"].to(device),
            batch_dict["p_mask"].to(device),
            batch_dict["p_point"].to(device),
            batch_dict["gt"].to(device),
        )
        outputs = model(target, examples, p_bboxes, p_masks, p_points)

        # outputs should be a tensor (N, M, H, W), where M is the number of classes
        # 0 is the background class
        metric_iou = JaccardIndex(task="multiclass", num_classes=outputs.shape[1])
        
        iou = metric_iou(outputs, gt)
        return iou

