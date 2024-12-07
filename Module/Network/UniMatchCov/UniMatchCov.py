import torch
import torch.utils

from pathlib import Path

from ..UniMatch.unimatch.unimatch import UniMatch
from ..UniMatch.unimatch.transformer import FeatureTransformer
from ..UniMatch.unimatch.utils import upsample_flow_with_mask

from ..CovHead import LinearCovHead, CNNCovHead


class UniMatchCov(torch.nn.Module):
    def __init__(self, fwd_kwargs: dict, num_scales=1, feature_channels=128, upsample_factor=8, num_head=1,
                ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=False, task='flow', cov_model="LinearCovHead"):
        super().__init__()
        self.unimatch = UniMatch(num_scales, feature_channels, upsample_factor, num_head, ffn_dim_expansion, num_transformer_layers, reg_refine, task)
        self.fwd_kwargs = fwd_kwargs
        self.unimatch.transformer.register_forward_hook(hook=self._hook)
        
        self.transformer = None

        match cov_model:
            case "LinearCovHead":
                self.cov_module = LinearCovHead()
            case "ConvCovHead":
                self.cov_module = CNNCovHead()
            case "FormerCovHead":
                self.cov_module = CNNCovHead()
                self.transformer = FeatureTransformer(num_layers=2,
                                    d_model=feature_channels,
                                    nhead=num_head,
                                    ffn_dim_expansion=ffn_dim_expansion)
            case _:
                raise ValueError(f"Unavailable cov model {cov_model}")

        self.upsampler = torch.nn.Sequential( # TODO: define a unified upsampler
            torch.nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0)
        )

    def cov_prediction(self, feature0: torch.Tensor, feature1: torch.Tensor):
        if self.transformer is not None:
            feature0, feature1 = self.transformer(feature0, feature1,  attn_type=self.fwd_kwargs["attn_type"], attn_num_splits=self.fwd_kwargs["attn_splits_list"][0]) ## TODO we are not supporting regrefine yet
        pred_cov = self.cov_module(feature0)  # [B, 2, H, W]
        pred_cov = self.upsample_flow(pred_cov, feature0, bilinear=False)

        return [pred_cov]
    
    def _hook(self, module: UniMatch, input, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.out_slot: list[torch.Tensor] = self.cov_prediction(output[0], output[1])

    def forward(self, img0: torch.Tensor, img1: torch.Tensor)-> dict[str, list[torch.Tensor]]:
        orig_output = self.unimatch(img0, img1, **self.fwd_kwargs)
        cov_output  = {"flow_cov_preds": self.out_slot}
        assert cov_output is not None
        return orig_output | cov_output
    
    def upsample_flow(self, flow: torch.Tensor, feature: torch.Tensor|None=None, bilinear=True, upsample_factor=8):
        if bilinear:
            up_flow = torch.nn.functional.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor
        else:
            assert feature is not None
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=upsample_factor)
        return up_flow
    
    @torch.inference_mode(mode=True)
    def inference(self, img0: torch.Tensor, img1: torch.Tensor):
        result = self(img0, img1)
        ### TODO: support the cov for a list of flow_preds
        flow, flow_cov = result["flow_preds"][-1], result["flow_cov_preds"]
        return flow, flow_cov
    
    def load_unimatch_ckpt(self, weight: Path):
        ckpt  = torch.load(weight, weights_only=True)
        self.unimatch.load_state_dict(ckpt["model"])
        
    def load_ckpt(self, weight: Path):
        ckpt = torch.load(weight, weights_only=True)
        self.load_state_dict(ckpt)