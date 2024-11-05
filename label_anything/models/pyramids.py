import torch
import torch.nn as nn
import math


class PyramidSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, d_model, positional_embedding_temperature):
        super().__init__()
        self.embedding_dim = d_model // 2
        self.temperature = positional_embedding_temperature
        self.scale = 2 * math.pi

    def forward(self, pixel_values, pixel_mask):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.embedding_dim, dtype=torch.float32, device=pixel_values.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PyramidNeck(nn.Module):
    def __init__(
        self,
        intermediate_channel_sizes,
        feature_levels=["stage2", "stage3", "stage4"],
        image_resolution=384,
        d_model=512,
        positional_embedding_temperature=20,
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.position_embedding = PyramidSinePositionEmbedding(
            d_model, positional_embedding_temperature
        )

        # Create input projection layers
        if len(feature_levels) > 1:
            num_backbone_outs = len(intermediate_channel_sizes)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = intermediate_channel_sizes[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                )
            for _ in range(len(feature_levels) - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, d_model, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, d_model),
                    )
                )
                in_channels = d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            intermediate_channel_sizes[-1], d_model, kernel_size=1
                        ),
                        nn.GroupNorm(32, d_model),
                    )
                ]
            )

        self.level_embed = nn.Parameter(torch.Tensor(len(feature_levels), d_model))
        self.final_conv = nn.Conv2d(
            len(feature_levels) * d_model, d_model, kernel_size=1
        )

    def forward(self, feature_pyramid):
        key0 = next(iter(feature_pyramid.keys()))
        batch_size = feature_pyramid[key0].shape[0]
        pixel_mask = torch.ones(
            ((batch_size, self.image_resolution, self.image_resolution)),
            dtype=torch.long,
            device=feature_pyramid[key0].device,
        )

        pyr = []
        for feature_map in feature_pyramid.values():
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(
                pixel_mask[None].float(), size=feature_map.shape[-2:]
            ).to(torch.bool)[0]
            pyr.append((feature_map, mask))

        pos = []
        for feature_map, mask in pyr:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(pyr):
            feature_maps.append(self.input_proj_vision[level](source))
            masks.append(mask)

        # add the level embeddings to the position embeddings
        for i in range(len(pos)):
            pos[i] += self.level_embed[i].view(1, -1, 1, 1)

        # add the position embeddings to the feature maps
        for i in range(len(feature_maps)):
            feature_maps[i] += pos[i]

        # apply pixel masks (it's useless but i keep it as in HF)
        for i in range(len(feature_maps)):
            feature_maps[i] = feature_maps[i] * masks[i].unsqueeze(1).to(
                feature_maps[i].dtype
            )

        # interpolate the feature maps to the size of the largest feature map (the first one)
        for i in range(1, len(feature_maps)):
            feature_maps[i] = nn.functional.interpolate(
                feature_maps[i],
                size=(feature_maps[0].shape[-2], feature_maps[0].shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        # make a single tensor by summing all the feature maps
        # feature_maps = torch.stack(feature_maps, dim=0).sum(dim=0)

        # concatenate the feature maps
        feature_maps = torch.cat(feature_maps, dim=1)
        feature_maps = self.final_conv(feature_maps)
        return feature_maps
