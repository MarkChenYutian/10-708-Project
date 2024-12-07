"""
This file contains a covariance estimation model
"""

import Module
import torch
from types import SimpleNamespace

from Module.Covariance.ObserveCov import LearningCovariance

from DataLoader import DataFramePair, StereoFrame
from Utility.Point import filterPointsInRange, pixel2point_NED 
from Utility.Extensions import ConfigTestable

from .Utils import TrainInstabilityException


class CovarianceAligner(ConfigTestable):
    def __init__(
        self,
        config: SimpleNamespace,
        frontend: Module.IFrontend,
        keypoint: Module.IKeypointSelector,
        ref_covmodel: Module.IObservationCov,
        learn_covmodel: LearningCovariance,
    ):
        self.config   = config
        self.frontend = frontend
        self.keypoint = keypoint
        self.ref_covmodel = ref_covmodel
        self.lrn_covmodel = learn_covmodel
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config.args, {
            "numPoints": lambda v: isinstance(v, int)
        })
    
    @classmethod
    def from_config(cls: type["CovarianceAligner"], odomcfg: SimpleNamespace) -> "CovarianceAligner":
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        ReferenceCovModel   = Module.IObservationCov.instantiate(odomcfg.ref_obscov.type, odomcfg.ref_obscov.args)
        ObserveCovModel     = Module.IObservationCov.instantiate(odomcfg.learn_obscov.type, odomcfg.learn_obscov.args)
        assert isinstance(ObserveCovModel, LearningCovariance)
        
        return cls(
            config   = odomcfg.args,
            frontend = Frontend,
            keypoint = KeypointSelector,
            ref_covmodel   = ReferenceCovModel,
            learn_covmodel = ObserveCovModel
        )
    
    def estimate(self, frame: DataFramePair[StereoFrame]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            depth0, _       = self.frontend.estimate(None            , frame.cur.stereo)
            depth1, match01 = self.frontend.estimate(frame.cur.stereo, frame.nxt.stereo)
        assert (depth0.cov is not None) and (depth1.cov is not None) and (match01.cov is not None)

         # Calculate frame 0 and related stuff
        keypoint_0    = self.keypoint.select_point(frame.cur.stereo, self.config.numPoints, depth0, match01)
        kp0_depth     = self.frontend.retrieve_pixels(keypoint_0, depth0.depth)[0]
        kp0_depth_cov = self.frontend.retrieve_pixels(keypoint_0, depth0.cov)[0]
        
        # 
        mask          = kp0_depth < frame.cur.stereo.frame_baseline * frame.cur.stereo.fx
        keypoint_0    = keypoint_0[mask]
        kp0_depth     = kp0_depth[mask]
        kp0_depth_cov = kp0_depth_cov[mask]
        # End
        
        if keypoint_0.size(0) <= 8:
            raise TrainInstabilityException(msg=f"Can't estimate uncertainty since little points are provided.")
        
        points0       = pixel2point_NED(keypoint_0, kp0_depth, frame.cur.stereo.frame_K)
        points0_ref_covTc = self.ref_covmodel.estimate(frame.cur.stereo, keypoint_0, depth0, None, kp0_depth_cov, None)
        points0_lrn_covTc = self.lrn_covmodel.estimate(frame.cur.stereo, keypoint_0, depth0, None, kp0_depth_cov, None)
        
        points0_ref_mask  = ~(points0_ref_covTc.isnan().any(dim=[-1, -2]) | points0_ref_covTc.isinf().any(dim=[-1, -2]))
        points0_lrn_mask  = ~(points0_lrn_covTc.isnan().any(dim=[-1, -2]) | points0_lrn_covTc.isinf().any(dim=[-1, -2]))
        points0_mask      = points0_ref_mask | points0_lrn_mask
        
        keypoint_0        = keypoint_0[points0_mask]
        points0           = points0[points0_mask]
        points0_ref_covTc = points0_ref_covTc[points0_mask]
        points0_lrn_covTc = points0_lrn_covTc[points0_mask]
        # End
        
        # Register frame 1 and related stuff
        keypoint_1    = keypoint_0 + self.frontend.retrieve_pixels(keypoint_0, match01.flow).T
        inbound_mask  = filterPointsInRange(
            keypoint_1,
            (self.config.edge_width, frame.nxt.stereo.width  - self.config.edge_width),
            (self.config.edge_width, frame.nxt.stereo.height - self.config.edge_width)
        )
        keypoint_1    = keypoint_1[inbound_mask]
        
        kp1_depth     = self.frontend.retrieve_pixels(keypoint_1, depth1.depth)[0]
        kp1_depth_cov = self.frontend.retrieve_pixels(keypoint_1, depth1.cov)[0]
        
        # 
        depth1_mask   = kp1_depth < frame.cur.stereo.frame_baseline * frame.cur.stereo.fx
        keypoint_1    = keypoint_1[depth1_mask]
        kp1_depth     = kp1_depth[depth1_mask]
        kp1_depth_cov = kp1_depth_cov[depth1_mask]
        # End
        
        kp1_match_cov = self.frontend.retrieve_pixels(keypoint_1, match01.cov).T
        points1       = pixel2point_NED(keypoint_1, kp1_depth, frame.nxt.stereo.frame_K)
        
        if keypoint_1.size(0) <= 8:
            raise TrainInstabilityException(msg=f"Can't estimate uncertainty since little points are provided.")
        
        points1_ref_covTc = self.ref_covmodel.estimate(frame.nxt.stereo, keypoint_1, depth1, match01, kp1_depth_cov, kp1_match_cov)
        points1_lrn_covTc = self.lrn_covmodel.estimate(frame.nxt.stereo, keypoint_1, depth1, match01, kp1_depth_cov, kp1_match_cov)
        
        points1_ref_mask  = ~(points1_ref_covTc.isnan().any(dim=[-1, -2]) | points1_ref_covTc.isinf().any(dim=[-1, -2]))
        points1_lrn_mask  = ~(points1_lrn_covTc.isnan().any(dim=[-1, -2]) | points1_lrn_covTc.isinf().any(dim=[-1, -2]))
        points1_mask      = points1_ref_mask | points1_lrn_mask
        
        points1           = points1[points1_mask]
        points1_ref_covTc = points1_ref_covTc[points1_mask]
        points1_lrn_covTc = points1_lrn_covTc[points1_mask]
        
        points0_lrn_covTc = points0_lrn_covTc[inbound_mask.to(points0_lrn_covTc.device)][depth1_mask.to(points0_lrn_covTc.device)][points1_mask]
        points0_ref_covTc = points0_ref_covTc[inbound_mask.to(points0_ref_covTc.device)][depth1_mask.to(points0_lrn_covTc.device)][points1_mask]
        
        #
        
        return points0_lrn_covTc, points1_lrn_covTc, points0_ref_covTc, points1_ref_covTc
