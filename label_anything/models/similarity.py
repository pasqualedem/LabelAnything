import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from label_anything.data.utils import BatchKeys, get_preprocess_shape


class SimilarityFewShotSegmenter(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        similarity: str = "cosine",
        image_size: int | None = None,
        custom_preprocess: bool = False,
        compare_size: int = None
    ):
        super().__init__()
        self.encoder = encoder
        self.similarity = similarity
        self.image_size = image_size
        self.custom_preprocess = custom_preprocess
        
        if compare_size is None and self.image_size is not None:
            self.compare_size = self.image_size
        self.compare_size = compare_size

        if similarity != "cosine":
            raise NotImplementedError(
                f"Similarity '{similarity}' not implemented. Only cosine is supported."
            )
            
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          original_size (torch.Tensor): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        max_original_size = torch.max(original_sizes.view(-1, 2), 0).values.tolist()
        original_sizes = original_sizes[:, 0, :]  # get real sizes of the query images
        input_sizes = [
            get_preprocess_shape(h, w, self.image_size) for (h, w) in original_sizes
        ]  # these are the input sizes without padding

        # interpolate masks to the model size
        masks = F.interpolate(
            masks,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.custom_preprocess:
            # remove padding from masks
            masks = [
                masks[i, :, : input_size[0], : input_size[1]]
                for i, input_size in enumerate(input_sizes)
            ]
            
        # interpolate masks to the original size
        masks = [
            F.interpolate(
                torch.unsqueeze(masks[i], 0),
                original_size.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for i, original_size in enumerate(original_sizes)
        ]

        # pad masks to the same size, use -inf so they don't affect the softmax
        masks = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_original_size[1] - original_sizes[i, 1],
                        0,
                        max_original_size[0] - original_sizes[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(masks)
            ]
        )

        # set padding to background class
        masks[:, 0, :, :][masks[:, 0, :, :] == float("-inf")] = 0
        return masks

    def forward(self, batch: dict) -> torch.Tensor:
        if BatchKeys.EMBEDDINGS in batch:
            embeddings = batch[BatchKeys.EMBEDDINGS]  # [B, M, D, H, W]
            B, M, D, H_e, W_e = embeddings.shape
        else:
            if self.encoder is None:
                raise ValueError(
                    "Encoder is None, but no embeddings provided in batch."
                )

            images = batch[BatchKeys.IMAGES]  # [B, M, C, H, W]
            B, M, C, H, W = images.shape
            images = rearrange(images, "b m c h w -> (b m) c h w")
            embeddings = self.encoder(images)  # [(B*M), D, H', W']
            D, H_e, W_e = embeddings.shape[1:]
            embeddings = rearrange(embeddings, "(b m) d h w -> b m d h w", b=B, m=M)

        # Resize to compare size
        if self.compare_size is not None:
            embeddings = rearrange(embeddings, "b m d h w -> (b m) d h w")
            embeddings = F.interpolate(
                embeddings,
                size=(self.compare_size, self.compare_size),
                mode="bicubic",
                align_corners=False,
            )
            embeddings = rearrange(embeddings, "(b m) d h w -> b m d h w", b=B, m=M)
            compare_size = self.compare_size
        else:
            compare_size = embeddings.shape[-1]

        # Split query and support
        query_embeddings = embeddings[:, 0]  # [B, D, H, W]
        support_embeddings = embeddings[:, 1:]  # [B, M-1, D, H, W]

        # Normalize for cosine similarity
        query_embeddings = F.normalize(query_embeddings, dim=1)
        support_embeddings = F.normalize(support_embeddings, dim=2)

        # Handle prompt masks
        prompt_masks = batch[BatchKeys.PROMPT_MASKS]  # [B, M-1, N, H, W]
        B, M1, N, Hm, Wm = prompt_masks.shape

        # Resize prompt masks to match embedding resolution
        prompt_masks = F.interpolate(
            prompt_masks.reshape(B * M1 * N, 1, Hm, Wm),
            size=(compare_size, compare_size),
            mode="nearest",
        ).reshape(B, M1, N, compare_size, compare_size)

        # Compute cosine similarity
        # query: [B, D, H, W] -> [B, H*W, D]
        q = rearrange(query_embeddings, "b d h w -> b (h w) d")

        # support: [B, M1, D, H, W] -> [B, M1, H*W, D]
        s = rearrange(support_embeddings, "b m d h w -> b m (h w) d")

        # similarity: [B, (H*W_q), M1, (H*W_s)]
        sim = torch.einsum("bqd, bmkd -> bqmk", q, s)

        # Mask per class
        # prompt_masks: [B, M1, N, H, W] -> [B, M1, N, H*W]
        pm = rearrange(prompt_masks, "b m n h w -> b m n (h w)")
        other_classes = pm[:, :, 1:, :]  # tutte le classi tranne background
        background_mask = (other_classes.sum(dim=2) == 0).float()  # [B, M1, H*W]
        pm[:, :, 0, :] = background_mask  # sostituisci classe 0 con il background corretto

        # Apply mask: for each query pixel q and class n, take max over all support pixels that belong to class n
        # sim: [B, Q, M1, K], pm: [B, M1, N, K]
        logits = []
        for n in range(N):
            class_mask = pm[:, :, n, :]  # [B, M1, K]
            class_mask = class_mask.unsqueeze(1)  # [B, 1, M1, K]

            masked_sim = sim.masked_fill(
                class_mask == 0, float("-inf")
            )  # [B, Q, M1, K]
            max_sim, _ = masked_sim.view(B, sim.shape[1], -1).max(dim=-1)  # [B, Q]

            logits.append(max_sim)

        logits = torch.stack(logits, dim=1)  # [B, N, Q]
        logits = rearrange(
            logits,
            "b n (h w) -> b n h w",
            h=query_embeddings.shape[2],
            w=query_embeddings.shape[3],
        )
        logits = self.postprocess_masks(logits, batch["dims"])

        return {"logits": logits}  # [B, N, H, W]


def build_similarity(
    encoder: torch.nn.Module = None,
    similarity: str = "cosine",
    image_size: int | None = None,
    custom_preprocess: bool = False,
    compare_size: int | None = None,
) -> SimilarityFewShotSegmenter:
    return SimilarityFewShotSegmenter(
        encoder=encoder, similarity=similarity, image_size=image_size, custom_preprocess=custom_preprocess, compare_size=compare_size
    )
