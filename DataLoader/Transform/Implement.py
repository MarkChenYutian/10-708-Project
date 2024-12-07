import torch
from typing import Literal, get_args, TypeVar
from types import SimpleNamespace

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, center_crop

from DataLoader.Interface import DataFrame, StereoFrame, StereoData
from .Interface import IDataTransform



def scale_stereo(self: StereoData, scale_u: float, scale_v: float, interpolate: Literal["nearest", "bilinear"]) -> StereoData:
    match interpolate:
        case "bilinear": interp = InterpolationMode.BILINEAR
        case "nearest" : interp = InterpolationMode.NEAREST_EXACT
    
    raw_height = self.height
    raw_width  = self.width
    
    target_h   = int(raw_height / scale_v)
    target_w   = int(raw_width  / scale_u)
    
    round_scale_v = raw_height / target_h
    round_scale_u = raw_width  / target_w
    
    self.K = self.K.clone()
    self.height = target_h
    self.width  = target_w
    self.K[:, 0] /= round_scale_u
    self.K[:, 1] /= round_scale_v
    
    self.imageL = resize(self.imageL, [target_h, target_w], interpolation=interp)
    self.imageR = resize(self.imageR, [target_h, target_w], interpolation=interp)
    
    if self.gt_flow is not None:
        self.gt_flow = resize(self.gt_flow, [target_h, target_w], interpolation=interp)
        self.gt_flow[:, 0] /= round_scale_u
        self.gt_flow[:, 1] /= round_scale_v
    
    if self.flow_mask is not None:
        self.flow_mask = resize(self.flow_mask, [target_h, target_w], interpolation=interp)
    
    if self.gt_depth is not None:
        self.gt_depth = resize(self.gt_depth, [target_h, target_w], interpolation=interp)
    
    return self


def crop_stereo(self: StereoData, target_h: int, target_w: int) -> StereoData:
        orig_h, orig_w = self.height, self.width
        self.imageL = center_crop(self.imageL, [target_h, target_w])
        self.imageR = center_crop(self.imageR, [target_h, target_w])
        
        if self.gt_flow is not None:
            self.gt_flow   = center_crop(self.gt_flow, [target_h, target_w])
        if self.flow_mask is not None:
            self.flow_mask = center_crop(self.flow_mask, [target_h, target_w])
        if self.gt_depth is not None:
            self.gt_depth  = center_crop(self.gt_depth, [target_h, target_w])
        
        self.K = self.K.clone()
        self.height = target_h
        self.width  = target_w
        self.K[:, 0, 2] -= (orig_w - target_w) / 2.
        self.K[:, 1, 2] -= (orig_h - target_h) / 2.
        
        return self


T_co = TypeVar("T_co", bound=DataFrame)

class NoTransform(IDataTransform[T_co, T_co]):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        return

    def __call__(self, frame: T_co) -> T_co:
        return frame


class ScaleFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Scale the image & ground truths on u and v direction and modify the camera intrinsic accordingly.
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "scale_u": lambda v: isinstance(v, (float, int)) and v > 0,
            "scale_v": lambda v: isinstance(v, (float, int)) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"}
        })

    def __call__(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo = scale_stereo(frame.stereo, scale_u=self.config.scale_u, scale_v=self.config.scale_v, interpolate=self.config.interp)
        return frame


class CenterCropFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Center crop the image and modify ground truth & camera intrinsic accordingly.
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0
        })
    
    def __call__(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo = crop_stereo(frame.stereo, target_h=self.config.height, target_w=self.config.width)
        return frame


class AddImageNoise(IDataTransform[StereoFrame, StereoFrame]):
    """
    Add noise to image color. Note that the `stdv` is on scale of [0-255] image instead of
    on the scale of [0-1]. (That is, we will divide stdv by 255 when applying noise on image)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "stdv": lambda v: isinstance(v, (int, float)) and v > 0
        })
    
    def __call__(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo.imageL = (frame.stereo.imageL + (self.config.stdv / 255) * torch.randn_like(frame.stereo.imageL)).clamp(0.0, 1.0)
        frame.stereo.imageR = (frame.stereo.imageR + (self.config.stdv / 255) * torch.randn_like(frame.stereo.imageR)).clamp(0.0, 1.0)
        return frame


class CastDataType(IDataTransform[StereoFrame, StereoFrame]):
    T_SUPPORT = Literal["fp16", "fp32", "bf16"]
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "dtype": lambda v: v in get_args(CastDataType.T_SUPPORT)
        })
    
    def __call__(self, frame: StereoFrame) -> StereoFrame:
        frame.stereo.imageL = frame.stereo.imageL.to(dtype=self.config.dtype)
        frame.stereo.imageR = frame.stereo.imageR.to(dtype=self.config.dtype)
        if frame.stereo.gt_flow is not None  : frame.stereo.gt_flow   = frame.stereo.gt_flow.to(dtype=self.config.dtype)
        if frame.stereo.gt_depth is not None : frame.stereo.gt_depth  = frame.stereo.gt_depth.to(dtype=self.config.dtype)
        if frame.stereo.flow_mask is not None: frame.stereo.flow_mask = frame.stereo.flow_mask.to(dtype=self.config.dtype)
        return frame


class SmartResizeFrame(IDataTransform[StereoFrame, StereoFrame]):
    """
    Automatically resize and crop the frame to target height and width to
    maximize the fov of resulted frame while achieving target shape.
    
    This process will maintein the aspect ratio of the image (i.e. the image 
    will not be stretched)
    """
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "height": lambda v: isinstance(v, int) and v > 0,
            "width": lambda v: isinstance(v, int) and v > 0,
            "interp" : lambda v: v in {"nearest", "bilinear"},
        })
    
    def __call__(self, frame: StereoFrame) -> StereoFrame:
        orig_height, orig_width = frame.stereo.height, frame.stereo.width
        targ_height, targ_width = self.config.height, self.config.width
        
        scale_factor = min(orig_height / targ_height, orig_width / targ_width)
        frame.stereo = scale_stereo(frame.stereo, scale_u=scale_factor, scale_v=scale_factor, interpolate=self.config.interp)
        frame.stereo = crop_stereo(frame.stereo, target_h=targ_height, target_w=targ_width)
        
        return frame
