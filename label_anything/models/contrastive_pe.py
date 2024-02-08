import torch
from label_anything.models.prompt_encoder import PromptImageEncoder
from label_anything.data.prompt_encoder_dataset import PromptEncoderBatchKeys


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

    def forward(self, data_dict):
        clip_embeddings = data_dict.pop(PromptEncoderBatchKeys.CLIP_EMBEDDINGS)
        return self.prompt_proj(self.prompt_encoder(data_dict)), self.clip_proj(clip_embeddings).mean(dim=1)
