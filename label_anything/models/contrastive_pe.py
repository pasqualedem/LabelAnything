import torch
from label_anything.models.prompt_encoder import PromptImageEncoder
from label_anything.data.utils import BatchKeys


class ContrastivePromptEncoder(torch.nn.Module):
    def __init__(
            self,
            prompt_encoder: PromptImageEncoder,
            hidden_size: int,
            pe_in_channels: int,
            clip_in_channels: int,
    ):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.pe_in_channels = pe_in_channels
        self.clip_in_channels = clip_in_channels
        self.hidden_size = hidden_size
        self.prompt_proj = torch.nn.Linear(self.pe_in_channels, self.hidden_size)
        self.clip_proj = torch.nn.Linear(self.clip_in_channels, self.hidden_size)

    def prepare_prompts(self, batched_input):
        if (
                "prompt_points" in batched_input
                and (batched_input["flag_points"] == 0).all().logical_not()
        ):
            points = (batched_input["prompt_points"], batched_input["flag_points"])
        else:
            points = None
        if (
                "prompt_bboxes" in batched_input
                and (batched_input["flag_bboxes"] == 0).all().logical_not()
        ):
            boxes = batched_input["prompt_bboxes"]
            box_flag = batched_input["flag_bboxes"]
            boxes = (boxes, box_flag)
        else:
            boxes = None
        if (
                "prompt_masks" in batched_input
                and (batched_input["flag_masks"] == 0).all().logical_not()
        ):
            masks = (batched_input["prompt_masks"], batched_input["flag_masks"])
        else:
            masks = None

        return points, boxes, masks

    def forward(self, data_dict):
        clip_embeddings = data_dict.pop(BatchKeys.CLIP_EMBEDDINGS)
        points, boxes, masks = self.prepare_prompts(data_dict)
        class_embeddings = self.prompt_encoder(image_embeddings=data_dict[BatchKeys.EMBEDDINGS],
                                               points=points,
                                               boxes=boxes,
                                               masks=masks,)
        class_proj = self.prompt_proj(class_embeddings)
        clip_proj = self.clip_proj(clip_embeddings).mean(dim=1)
        return class_proj, clip_proj
