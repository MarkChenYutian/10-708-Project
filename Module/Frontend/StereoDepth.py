from __future__ import annotations

import torch
from types import SimpleNamespace
from typing import TypeVar, Generic, TypedDict, overload, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import StereoData
from Utility.Utils import padTo
from Utility.Extensions import ConfigTestableSubclass

# Stereo Depth interface ###
# T_Context = The internal state of stereo depth estimator
T_Context = TypeVar("T_Context")


class IStereoDepth(ABC, Generic[T_Context], ConfigTestableSubclass):
    """
    Estimate dense depth map of current stereo image.
    
    `IStereoDepth.estimate(frame: StereoData) -> IStereoDepth.Output`

    Given a frame with imageL, imageR being Bx3xHxW, return `output` where    

    * depth         - Bx1xHxW shaped torch.Tensor, estimated depth map
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    * cov           - Bx1xHxW shaped torch.Tensor or None, estimated covariance of depth map (if provided)
                    maybe padded with `nan` if model can't output prediction with same shape as input image.
    * mask          - Bx1xHxW shaped torch.Tensor or None, the position with `True` value means valid (not occluded) 
                    prediction regions.
    """
    @dataclass
    class Output:
        depth: torch.Tensor                    # torch.Tensor of shape B x 1 x H x W
        cov  : torch.Tensor | None = None      # torch.Tensor of shape B x 1 x H x W OR None if not applicable
        mask : torch.Tensor | None = None      # torch.Tensor of shape B x 1 x H x W OR None if not applicable
    
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
        self.context: T_Context       = self.init_context()
    
    @property
    @abstractmethod
    def provide_cov(self) -> bool: ...
    
    @abstractmethod
    def init_context(self) -> T_Context: ...
    
    @abstractmethod
    def estimate(self, frame: StereoData) -> Output: ...

    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map (Bx1xHxW).
        
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


# Stereo Depth Implementation ###
# Contexts

class ModelContext(TypedDict):
    model: torch.nn.Module

# Implementations

class GTDepth(IStereoDepth[None]):
    """
    Always returns the ground truth depth. input frame must have `gtDepth` attribute non-empty.
    """
    @property
    def provide_cov(self) -> bool: return False
    
    def init_context(self) -> None: return None
    
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        assert frame.gt_depth is not None
        gt_depthmap = padTo(frame.gt_depth, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
        
        return IStereoDepth.Output(depth=gt_depthmap)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class FlowFormerDepth(IStereoDepth[ModelContext]):
    """
    Use FlowFormer to estimate disparity between rectified stereo image. Does not generate depth_cov.
    
    See FlowFormerCovDepth for jointly estimating depth and depth_cov
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
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        est_flow, _ = self.context["model"].inference(frame.imageL, frame.imageR)
        disparity = est_flow[:1].abs().unsqueeze(0)
        depth_map = ((frame.frame_baseline * frame.fx) / disparity)
        
        # return depth_map, None
        return IStereoDepth.Output(depth=depth_map)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class FlowFormerCovDepth(IStereoDepth[ModelContext]):
    """
    Use modified FlowFormer to estimate diparity between rectified stereo image and uncertainty of disparity.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> ModelContext:
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        model = build_flowformer(get_cfg(), self.config.device)
        ckpt  = torch.load(self.config.weight)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
        
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        est_flow, est_cov = self.context["model"].inference(frame.imageL, frame.imageR)
        disparity, disparity_cov = est_flow[:, :1].abs(), est_cov[:, :1]
        
        # Propagate disparity covariance to depth covariance
        # See Appendix A.1 of the MAC-VO paper
        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_map = ((frame.frame_baseline * frame.fx) / disparity)
        depth_cov = (((frame.frame_baseline * frame.fx) ** 2) * (error_rate_2 / disparity_2))
        
        return IStereoDepth.Output(depth=depth_map, cov=depth_cov)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            })


class TartanVODepth(IStereoDepth[ModelContext]):
    """
    Use the StereoNet from TartanVO to estimate diparity between stereo image. 
    
    Does not estimate depth_cov if config.cov_mode set to 'None'.
    where config.cov_mode = {'None', 'Est'}
    """
    @property
    def provide_cov(self) -> bool: return self.config.cov_mode == "Est"
    
    def init_context(self) -> ModelContext:
        from Utility.Config import build_dynamic_config
        from Module.Network.StereoCov import StereoCovNet
        
        cfg, _ = build_dynamic_config({"exp": False, "decoder": "hourglass"})
        model = StereoCovNet(cfg)
        ckpt = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        
        model.eval()
        return ModelContext(model=model)
        
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        depth, depth_cov = self.context["model"].inference(frame)
        
        depth_map = padTo(depth, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
        
        if self.config.cov_mode == "Est":
            depth_cov = padTo(depth_cov, (frame.height, frame.width), dim=(-2, -1), value=float('nan'))
            return IStereoDepth.Output(depth=depth_map, cov=depth_cov)
        else:
            return IStereoDepth.Output(depth=depth_map)
        
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
                "weight"    : lambda s: isinstance(s, str),
                "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
                "cov_mode"  : lambda s: s in {"Est", "None"}
            })


class UniMatchStereoDepth(IStereoDepth[ModelContext]):
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
            "attn_type": lambda s: s in ("swin", "self_swin2d_cross_swin1d"),
            "attn_splits_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            "corr_radius_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            "prop_radius_list": lambda arr: isinstance(arr, list) and all(map(lambda v: isinstance(v, int), arr)),
            
            "weight": lambda s: isinstance(s, str),
            "device": lambda s: isinstance(s, str) and (("cuda" in s) or s == "cpu")
        })
    
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        results = self.context["model"](
            frame.imageL.to(self.config.device) * 255.,
            frame.imageR.to(self.config.device) * 255.,
            attn_type=self.config.attn_type,
            attn_splits_list=self.config.attn_splits_list,
            corr_radius_list=self.config.corr_radius_list,
            prop_radius_list=self.config.prop_radius_list,
            task=self.config.task
        )
        disparity: torch.Tensor = results['flow_preds'][-1].unsqueeze(1)
        return IStereoDepth.Output(depth=disparity.reciprocal() * (frame.frame_baseline * frame.fx))


class UniMatchCovDepth(IStereoDepth[ModelContext]):
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

    # def estimate(self, frame: StereoData) -> tuple[torch.Tensor, torch.Tensor | None]:
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        results = self.context["model"].inference(
            frame.imageL.to(self.config.device) * 255.,
            frame.imageR.to(self.config.device) * 255.,
        )
        est_flow, est_cov = results
        disparity, disparity_cov = est_flow[:, :1].abs(), est_cov[-1][:, :1]

        # Propagate disparity covariance to depth covariance
        # See Appendix A.1 of paper
        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_map = ((frame.frame_baseline * frame.fx) / disparity)
        depth_cov = (((frame.frame_baseline * frame.fx) ** 2) * (error_rate_2 / disparity_2))

        return IStereoDepth.Output(depth=depth_map, cov=depth_cov)




class UniCeptionDepth(IStereoDepth[ModelContext]):
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
    
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        if self.image_resizer is not None:
            image0 = (frame.imageL.to(self.config.device) * 255.).to(torch.uint8)
            image1 = (frame.imageR.to(self.config.device) * 255.).to(torch.uint8)
            image0, image1, ctx = self.crop_to_shape(image0, image1)
            image0 = ((image0.float() / 255.) - self.image_mean) / self.image_std
            image1 = ((image1.float() / 255.) - self.image_mean) / self.image_std
        else:
            ctx = None
            image0 = (frame.imageL.to(self.config.device) - self.image_mean) / self.image_std
            image1 = (frame.imageR.to(self.config.device) - self.image_mean) / self.image_std
        
        view1 = {"img": image0, "instance": [0], "data_norm_type": self.data_norm_type}
        view2 = {"img": image1, "instance": [1], "data_norm_type": self.data_norm_type}
        
        with torch.no_grad(), torch.autocast(self.config.device, torch.bfloat16):
            result = self.context["model"](view1, view2)
        
        flow           = result["flow"]["flow_output"]
        flow_cov       = result["flow"]["flow_covariance"]
        occlusion_mask = result["occlusion"]["mask"]
        
        if self.image_resizer is not None:
            assert self.flow_unmap and self.cov_unmap and self.mask_unmap and ctx
            
            image0_src_shape = (frame.imageL.size(-2), frame.imageL.size(-1))
            image1_src_shape = (frame.imageR.size(-2), frame.imageR.size(-1))
        
            flow, flow_valid = self.flow_unmap(flow, *ctx, img0_source_shape=image0_src_shape, img1_source_shape=image1_src_shape)
            flow_cov, flow_cov_valid = self.cov_unmap(flow_cov, ctx[0], ctx[2], img0_source_shape=image0_src_shape)
            occlusion_mask, occ_mask_valid = self.mask_unmap(occlusion_mask, ctx[0], ctx[2], img0_source_shape=image0_src_shape)
        occlusion_mask = occlusion_mask >= 0.5
        
        disparity, disparity_cov = flow[:, :1].abs(), flow_cov[:, :1]
        
        # Propagate disparity covariance to depth covariance
        # See Appendix A.1 of the MAC-VO paper
        disparity_2 = disparity.square()
        error_rate_2 = disparity_cov / disparity_2
        depth_map  = ((frame.frame_baseline * frame.fx) / disparity)
        depth_cov  = (((frame.frame_baseline * frame.fx) ** 2) * (error_rate_2 / disparity_2))
        depth_mask = occlusion_mask[..., :1]
        
        return IStereoDepth.Output(depth=depth_map, cov=depth_cov, mask=depth_mask)


# Modifier - modifies the input/output of another estimator
# Modifier(IStereoDepth) -> IStereoDepth'


class ApplyGTDepthCov(IStereoDepth[IStereoDepth]):
    """
    A higher-order-module that encapsulates a IStereoDepth module. 
    
    Always compare the estimated output of encapsulated IStereoDepth with ground truth depth and convert
    error in estimation to 'estimated' covariance.
    
    Will raise AssertionError if frame does not have gtDepth.
    """
    @property
    def provide_cov(self) -> bool: return True
    
    def init_context(self) -> IStereoDepth:
        internal_module = IStereoDepth.instantiate(self.config.module.type, self.config.module.args)
        return internal_module
    
    @torch.inference_mode()
    def estimate(self, frame: StereoData) -> IStereoDepth.Output:
        assert frame.gt_depth is not None
        
        output = self.context.estimate(frame)
        error = frame.gt_depth.to(output.depth.device) - output.depth
        gt_cov = error.square()
        
        output.cov = gt_cov
        return output

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IStereoDepth.is_valid_config(config.module)
