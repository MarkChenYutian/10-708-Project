from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass

from DataLoader import StereoData
from Utility.Utils import padTo
from Utility.Extensions import ConfigTestableSubclass
from Utility.Config import build_dynamic_config

# Matching interface ###
# T_Context = The internal state of matcher
T_Context = TypeVar("T_Context")


class IMatcher(ABC, Generic[T_Context], ConfigTestableSubclass):
    @dataclass
    class Output:
        flow: torch.Tensor                 # B x 2 x H x W, float32
        cov : torch.Tensor | None = None   # B x 3 x H x W, float32 OR None if not applicable
        mask: torch.Tensor | None = None   # B x 1 x H x W, bool    OR None if not applicable
    
        @property
        def as_full_cov(self) -> "IMatcher.Output":
            if self.cov is None or self.cov.size(1) == 3: return self
            B, C, H, W = self.cov.shape
            assert C == 2, f"number of channel for cov must be either 2 or 3, get {C=}"
            return IMatcher.Output(
                flow=self.flow,
                cov =torch.cat([self.cov, torch.zeros((B, 1, H, W), device=self.cov.device, dtype=self.cov.dtype)], dim=1),
                mask=self.mask
            )
        
    """
    Estimate the optical flow map between two frames. (Use left-frame of stereo pair)
    
    `IMatcher(frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * flow      - Bx2xHxW shaped torch.Tensor, estimated optical flow map
                maybe padded with `nan` if model can't output prediction with same shape as input image.
    * cov       - Bx3xHxW shaped torch.Tensor or None, estimated covariance of optical flow map map (if provided)
                maybe padded with `nan` if model can't output prediction with same shape as input image.
                The three channels are uu, vv, and uv respectively, such that the 2x2 covariance matrix will be:
                    Sigma = [[uu, uv], [uv, vv]]
    * mask      - Bx1xHxW shaped torch.Tensor or None, the position with `True` value means valid (not occluded) 
                prediction regions.
    """
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
        self.context: T_Context       = self.init_context()
    
    @property
    @abstractmethod
    def provide_cov(self) -> bool: ...
    
    @abstractmethod
    def init_context(self) -> T_Context: ...
    
    @abstractmethod
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output: ...
    
    def __call__(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        return self.estimate(frame_t1, frame_t2)

    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (CxN) from scalar_map (BxCxHxW).
        
        #### Note that the pixel_uv is in (x, y) format, not (row, col) format.
        
        #### Note that only first sample of scalar_map is used. (Batch idx=0)
        """
        if scalar_map is None: return None
        
        if interpolate:
            raise NotImplementedError("Not implemented yet")
        else:
            values = scalar_map[0, ..., pixel_uv[..., 1].long(), pixel_uv[..., 0].long()]
            return values

# End #######################

# Dense Matching Implementation ###
# Contexts

class ModelContext(TypedDict):
    model: torch.nn.Module

# Implementations

class GTMatcher(IMatcher[None]):
    """
    A matcher that returns ground truth optical flow.
    
    Will raise AssertionError if ground truth optical flow is not provided.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> None: return None
    
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        
        gt_flow = padTo(frame_t1.gt_flow, (frame_t1.height, frame_t1.width), (-2, -1), float('nan'))
        return IMatcher.Output(flow=gt_flow)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerMatcher(IMatcher[ModelContext]):
    """
    Use FlowFormer to estimate optical flow betweeen two frames. Does not generate match_cov.
    
    See FlowFormerCovMatcher for jointly estimating depth and match_cov
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormer.core.FlowFormer import build_flowformer
        
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
    
    @torch.inference_mode()
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, _ = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return IMatcher.Output(flow=flow.unsqueeze(0))
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class FlowFormerCovMatcher(IMatcher[ModelContext]):
    """
    Use the modified FlowFormer proposed by us to jointly estimate optical flow betweeen two frames.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)

    @torch.inference_mode()
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        return IMatcher.Output(flow=flow, cov=flow_cov).as_full_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVOMatcher(IMatcher[ModelContext]):
    """
    Use TartanVO to estimate optical flow between two frames.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> ModelContext:
        from ..Network.TartanVOStereo.StereoVO_Interface import TartanStereoVOMatch
        model = TartanStereoVOMatch(self.config.weight, True, self.config.device)
        return ModelContext(model=model)    #type: ignore

    @torch.inference_mode()
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow = self.context["model"].inference(frame_t1, frame_t1.imageL, frame_t2.imageL)
        
        mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
        pad_height = (frame_t1.height - flow.size(-2)) // 2
        pad_width  = (frame_t1.width  - flow.size(-1)) // 2
        mask[..., pad_height:-pad_height, pad_width:-pad_width] = True
        
        flow = padTo(flow, (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        
        return IMatcher.Output(flow=flow, mask=mask)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVOCovMatcher(IMatcher[ModelContext]):
    """
    Use a modified version of TartanVO frontend network to jointly estimate optical flow
    and its covariance between two frames.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from Module.Network.PWCNet import RAFTFlowCovNet
        cfg, _ = build_dynamic_config({
            "decoder": "raft", "dim": 64, "dropout": 0.1,
            "num_heads": 4, "mixtures": 4, "gru_iters": 12, "kernel_size": 3,
        })
        ckpt = torch.load(self.config.weight, map_location="cpu")
        model = RAFTFlowCovNet(cfg, self.config.device)
        model.load_ddp_state_dict(ckpt)

        model.eval()
        model = model.to(self.config.device)
        return ModelContext(model=model)
    
    @torch.inference_mode()
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.context["model"].inference(frame_t1.imageL, frame_t2.imageL)
        
        mask = torch.zeros_like(flow[:, :1], dtype=torch.bool)
        pad_height = (frame_t1.height - flow.size(-2)) // 2
        pad_width  = (frame_t1.width  - flow.size(-1)) // 2
        mask[..., pad_height:-pad_height, pad_width:-pad_width] = True
        
        flow     = padTo(flow    , (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        flow_cov = padTo(flow_cov, (frame_t1.height, frame_t1.width), dim=(-2, -1), value=float('nan'))
        return IMatcher.Output(flow=flow, cov=flow_cov, mask=mask).as_full_cov

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:    
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class GMFlowMatcher(IMatcher[ModelContext]):
    @property
    def provide_cov(self) -> bool: return False
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight": lambda s: isinstance(s, str),
            "device": lambda s: isinstance(s, str) and (("cuda" in s) or s == "cpu")
        })
    
    def init_context(self) -> ModelContext:
        from ..Network.GMFlow.gmflow.gmflow import GMFlow
        model = GMFlow(
            num_scales=1, upsample_factor=8, feature_channels=128,
            attention_type='swin', num_transformer_layers=6, ffn_dim_expansion=4,
            num_head=1
        ).to(self.config.device)
        ckpt  = torch.load(self.config.weight, weights_only=True, map_location=self.config.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return ModelContext(model=model)

    # def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[torch.Tensor, torch.Tensor | None]:
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        results = self.context["model"](
            (frame_t1.imageL.to(self.config.device) * 255.),
            (frame_t2.imageL.to(self.config.device) * 255.),
            [2], [-1], [-1], False
        )
        return IMatcher.Output(flow=results["flow_preds"][-1])


class UniMatchMatcher(IMatcher[ModelContext]):
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> ModelContext:
        from ..Network.UniMatch.unimatch.unimatch import UniMatch
        model = UniMatch(
            num_scales          =self.config.num_scales,
            feature_channels    =self.config.feature_channels,
            upsample_factor     =self.config.upsample_factor,
            num_head            =self.config.num_head,
            ffn_dim_expansion   =self.config.ffn_dim_expansion,
            num_transformer_layers=self.config.num_transformer_layer,
            reg_refine          =self.config.num_refine > 0,
            task                =self.config.task
        ).to(self.config.device)
        ckpt = torch.load(self.config.weight, weights_only=True, map_location=self.config.device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        return ModelContext(model=model)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "num_scales": lambda v: isinstance(v, int) and v >= 0,
            "feature_channels": lambda v: isinstance(v, int) and v > 0,
            "upsample_factor": lambda v: v in (4, 8),
            "num_head": lambda v: isinstance(v, int) and v > 0,
            "ffn_dim_expansion": lambda v: isinstance(v, int) and v > 0,
            "num_transformer_layer": lambda v: isinstance(v, int) and v > 0,
            "num_refine": lambda v: isinstance(v, int) and v >= 0,
            "task": lambda s: isinstance(s, str) and s in ("flow", "stereo", "depth"),
            "attn_type": lambda s: s == "swin",
            "attn_splits_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            "corr_radius_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            "prop_radius_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            
            "weight": lambda s: isinstance(s, str),
            "device": lambda s: isinstance(s, str) and (("cuda" in s) or s == "cpu")
        })

    # def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[torch.Tensor, torch.Tensor | None]:
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        results = self.context["model"](
            frame_t1.imageL.to(self.config.device) * 255.,
            frame_t2.imageL.to(self.config.device) * 255.,
            attn_type=self.config.attn_type,
            attn_splits_list=self.config.attn_splits_list,
            corr_radius_list=self.config.corr_radius_list,
            prop_radius_list=self.config.prop_radius_list,
            task=self.config.task
        )
        return IMatcher.Output(flow=results["flow_preds"][-1]).as_full_cov


class UniMatchCovMatcher(IMatcher[ModelContext]):
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from ..Network.UniMatchCov import UniMatchCov
        model = UniMatchCov(
            fwd_kwargs=vars(self.config.model.fwd_args),
            **vars(self.config.model.args)
        ).to(self.config.device)
        ckpt = torch.load(self.config.weight, weights_only=True, map_location=self.config.device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        model.eval()
        return ModelContext(model=model)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight": lambda s: isinstance(s, str),
            "device": lambda s: isinstance(s, str) and (("cuda" in s) or s == "cpu")
        })

    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        flow, flow_cov = self.context["model"].inference(
            frame_t1.imageL.to(self.config.device) * 255.,
            frame_t2.imageL.to(self.config.device) * 255.,
        )
        return IMatcher.Output(flow=flow, cov=flow_cov[-1]).as_full_cov


class UniCeptionMatcher(IMatcher[ModelContext]):
    """
    Use the UniCeption codebase for optical flow estimation
    """
    def __init__(self, config: SimpleNamespace):
        from Utility.MatchAnything.FlowResize import AutomaticShapeSelection, ResizeToFixedManipulation, unmap_predicted_flow, unmap_predicted_channels, unmap_covariance
        self.image_mean: torch.Tensor | None = None
        self.image_std : torch.Tensor | None = None
        self.data_norm_type: str | None = None
        
        # Resizing strategy
        self.resolution    = tuple(config.resolution) if config.resolution[0] > -1 else None
        if self.resolution is not None:
            self.image_resizer = AutomaticShapeSelection(ResizeToFixedManipulation(self.resolution))
            self.flow_unmap    = unmap_predicted_flow
            self.cov_unmap     = unmap_covariance
            self.mask_unmap    = unmap_predicted_channels
        else:
            self.image_resizer = None
            self.flow_unmap    = None
            self.cov_unmap     = None
            self.mask_unmap    = None
        
        super().__init__(config)
    
    @property
    def provide_cov(self) -> bool: return False
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight"    : lambda s: isinstance(s, str),
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "resolution": lambda arr: isinstance(arr, list) and isinstance(arr[0], int) and isinstance(arr[1], int) and len(arr) == 2
        })
    
    def crop_to_shape(self, image0: torch.Tensor, image1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
        assert self.image_resizer is not None
        (
            scaled_img0, scaled_img1,
            img0_region_source, img1_region_source,
            img0_region_representation, img1_region_representation,
        ) = self.image_resizer(image0.permute(0, 2, 3, 1), image1.permute(0, 2, 3, 1))
        
        resize_context = (img0_region_representation, img1_region_representation, img0_region_source, img1_region_source)
        return scaled_img0.permute(0, 3, 1, 2), scaled_img1.permute(0, 3, 1, 2), resize_context
    
    def init_context(self) -> ModelContext:
        from Utility.PrettyPrint import Logger
        from Module.Network.UniCeption.uniception.models.factory import MatchAnythingModel
        from Module.Network.UniCeption.uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        
        Logger.write("info", f"Loading model from {self.config.weight}")
        model  = MatchAnythingModel.from_pretrained(self.config.weight, strict=False)
        model.to(self.config.device)
        model.eval()
        
        image_norm = IMAGE_NORMALIZATION_DICT[model.encoder.data_norm_type]
        self.data_norm_type = model.encoder.data_norm_type
        self.image_mean = image_norm.mean.to(self.config.device).view(1, 3, 1, 1)
        self.image_std  = image_norm.std .to(self.config.device).view(1, 3, 1, 1)
        
        return {"model": model}
    
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        if self.image_resizer is not None:
            image0 = (frame_t1.imageL.to(self.config.device) * 255.).to(torch.uint8)
            image1 = (frame_t2.imageL.to(self.config.device) * 255.).to(torch.uint8)
            image0, image1, ctx = self.crop_to_shape(image0, image1)
            image0 = ((image0.float() / 255.) - self.image_mean) / self.image_std
            image1 = ((image1.float() / 255.) - self.image_mean) / self.image_std
        else:
            ctx = None
            image0 = (frame_t1.imageL.to(self.config.device) - self.image_mean) / self.image_std
            image1 = (frame_t2.imageL.to(self.config.device) - self.image_mean) / self.image_std
        
        view1 = {"img": image0, "instance": [0], "data_norm_type": self.data_norm_type}
        view2 = {"img": image1, "instance": [1], "data_norm_type": self.data_norm_type}
        
        with torch.no_grad(), torch.autocast(self.config.device, torch.bfloat16):
            result = self.context["model"](view1, view2)
        
        flow           = result["flow"]["flow_output"]
        flow_cov       = result["flow"]["flow_covariance"]
        occlusion_mask = result["occlusion"]["mask"]
        
        if self.image_resizer is not None:
            assert self.flow_unmap and self.cov_unmap and self.mask_unmap and ctx
            
            image0_src_shape = (frame_t1.imageL.size(-2), frame_t1.imageL.size(-1))
            image1_src_shape = (frame_t2.imageL.size(-2), frame_t2.imageL.size(-1))
        
            flow_up, flow_valid = self.flow_unmap(flow, *ctx, img0_source_shape=image0_src_shape, img1_source_shape=image1_src_shape)
            flow_cov_up, flow_cov_valid = self.cov_unmap(flow_cov, ctx[0], ctx[2], img0_source_shape=image0_src_shape)
            occ_mask_up, occ_mask_valid = self.mask_unmap(occlusion_mask, ctx[0], ctx[2], img0_source_shape=image0_src_shape)
            occ_mask_up = occ_mask_up >= 0.5
            return IMatcher.Output(flow=flow_up, cov=flow_cov_up, mask=occ_mask_up)
        else:
            return IMatcher.Output(flow=flow, cov=flow_cov, mask=occlusion_mask)


# Modifier
# Modifier(IMatcher) -> IMatcher'


class ApplyGTMatchCov(IMatcher[IMatcher]):
    """
    A higher-order-module that encapsulates a IMatcher module. 
    
    Always compare the estimated output of encapsulated IMatcher with ground truth matching and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtFlow.
    
    NOTE: This modifier only creates estimation to Sigma matrix as a diagonal form, since the optimum 
    covariance matrix (that maximize log-likelihood of ground truth flow) is degenerated for a full
    2x2 matrix setup.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> IMatcher:
        internal_module = IMatcher.instantiate(self.config.module.type, self.config.module.args)
        return internal_module
    
    def estimate(self, frame_t1: StereoData, frame_t2: StereoData) -> IMatcher.Output:
        assert frame_t1.gt_flow is not None
        out = self.context.estimate(frame_t1, frame_t2)
        
        flow_error = out.flow - frame_t1.gt_flow.to(out.flow.device)
        flow_cov   = flow_error.square()
        out.cov = flow_cov
        return out.as_full_cov
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.module)
