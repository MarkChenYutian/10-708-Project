"""
This file contains a simplified version of MAC-VO that is specifically used for training
the attatched covariance modifier.
"""
import Module
import torch
import pypose as pp
from types import SimpleNamespace

from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.solver import PINV

from Module.Covariance.ObserveCov import Modifier_Learning, LearningCovariance

from DataLoader import DataFramePair, StereoFrame, StereoData
from Utility.Point import filterPointsInRange, pixel2point_NED 
from Utility.Extensions import ConfigTestable
from Utility.Math import MahalanobisDist

from .BLO.GNOptimizer import GaussNewton2
from .Utils import TrainInstabilityException


class PoseGraph(torch.nn.Module):
    def __init__(self, init_pose: pp.LieTensor, points_Tc: torch.Tensor, points_Tw: torch.Tensor, cov_Tc: torch.Tensor, cov_Tw: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("points_Tc", points_Tc)
        self.register_buffer("points_Tw", points_Tw)
        self.register_buffer("obs_covTc", cov_Tc)
        self.register_buffer("pts_covTw", cov_Tw)
        self.pose2opt = pp.Parameter(init_pose)

    def forward(self) -> torch.Tensor:
        return self.pose2opt.Act(self.points_Tc) - self.points_Tw
    
    def covariance(self) -> torch.Tensor:
        R = self.pose2opt.rotation().matrix()
        return (R @ self.obs_covTc @ R.T) + self.pts_covTw
    
    def error(self, pose: pp.LieTensor):
        err = pose.Act(self.points_Tc) - self.points_Tw
        return MahalanobisDist(err, torch.zeros_like(err), self.covariance())


class MACVO_Online(ConfigTestable):
    def __init__(
        self,
        config: SimpleNamespace,
        frontend: Module.IFrontend,
        keypoint: Module.IKeypointSelector,
        covmodel: Module.IObservationCov,
    ):
        assert isinstance(covmodel, (Modifier_Learning, LearningCovariance)), "Can only train covariance modifier under subclass of Modifier_Learning"
        self.config   = config
        self.frontend = frontend
        self.keypoint = keypoint
        self.covmodel = covmodel
        
        self.optimizer_ctx = {
            "kernel"   : (kernel:=Huber(delta=2.0)),
            "solver"   : PINV(),
            "corrector": FastTriggs(kernel),
            "vectorize": config.optim.vectorize,
            "device"   : config.optim.device
        }
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        cls._enforce_config_spec(config.args, {
            "numPoints": lambda v: isinstance(v, int)
        })
    
    @classmethod
    def from_config(cls: type["MACVO_Online"], odomcfg: SimpleNamespace) -> "MACVO_Online":
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        ObserveCovModel     = Module.IObservationCov.instantiate(odomcfg.obscov.type, odomcfg.obscov.args)
        
        return cls(
            config   = odomcfg.args,
            frontend = Frontend,
            keypoint = KeypointSelector,
            covmodel = ObserveCovModel
        )
    
    
    def estimate_sample(self, frame: DataFramePair[StereoFrame], 
                        depth0: Module.IStereoDepth.Output, 
                        depth1: Module.IStereoDepth.Output, 
                        match01:Module.IMatcher.Output, 
                        optimizer_niter: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        points0_covTc = self.covmodel.estimate(frame.cur.stereo, keypoint_0, depth0, None, kp0_depth_cov, None).to(self.config.optim.device)
        
        points0_mask  = ~(points0_covTc.isnan().any(dim=[-1, -2]) | points0_covTc.isinf().any(dim=[-1, -2]))
        keypoint_0    = keypoint_0[points0_mask]
        points0       = points0[points0_mask]
        points0_covTc = points0_covTc[points0_mask]
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
        
        points1_covTc = self.covmodel.estimate(frame.nxt.stereo, keypoint_1, depth1, match01, kp1_depth_cov, kp1_match_cov).to(self.config.optim.device)
        
        points1_mask  = ~(points1_covTc.isnan().any(dim=[-1, -2]) | points1_covTc.isinf().any(dim=[-1, -2]))
        points1       = points1[points1_mask]
        points1_covTc = points1_covTc[points1_mask]
        points0       = points0[inbound_mask][depth1_mask][points1_mask]
        points0_covTc = points0_covTc[inbound_mask.to(self.config.optim.device)][depth1_mask.to(self.config.optim.device)][points1_mask.to(self.config.optim.device)]
        
        if points0_covTc.size(0) <= 8:
            raise TrainInstabilityException(msg=f"Can't run optimizer since less than 8 points are provided.")
        # End
        
        # Start optimization
        graph = PoseGraph(pp.identity_SE3(), points1, points0, points1_covTc, points0_covTc)
        graph = graph.double().to(self.config.optim.device)
        
        optim_2 = GaussNewton2(graph, solver=self.optimizer_ctx["solver"], 
                                kernel=self.optimizer_ctx["kernel"], corrector=self.optimizer_ctx["corrector"], 
                                vectorize=True)
        
        assert (frame.cur.gt_pose is not None) and (frame.nxt.gt_pose is not None)
        gt_motion  = (frame.cur.gt_pose.Inv() @ frame.nxt.gt_pose).Log().to(self.config.optim.device)
        
        # NOTE: pose_diffgraph is a variable that will be used to keep track of the 
        #       second-order optimization's update while mainteining a autodiff graph.
        pose_diffgraph = pp.identity_SE3(requires_grad=True, device=self.config.optim.device, dtype=torch.float64)
        
        for _ in range(optimizer_niter):
            weight: torch.Tensor = torch.block_diag(*torch.pinverse(graph.covariance()))
            loss, updates = optim_2.update(input=(), weight=weight)
            pose_diffgraph = updates[0][0] * pose_diffgraph
        
        # Supervised Training
        # By here, pose_diffgraph should have exactly the same value as graph.pose2opt
        se3_error = (gt_motion - pose_diffgraph.Log()).norm()
        rte_error = (gt_motion.translation() - pose_diffgraph.translation()).norm()
        roe_error = (gt_motion.rotation().Log() - pose_diffgraph.rotation().Log()).norm() * (180. / torch.pi)
        # End
        
        # Self-supervise training
        # error = torch.tensor(0., requires_grad=True, device=self.config.optim.device)
        # for update in updates:
        #     error = error + graph.error(update[-1]).sum()
        # End
        
        # Graph Residue
        # residue = graph.error(pose_diffgraph)
        # End
        
        return se3_error, rte_error, roe_error    # error
    
    
    @staticmethod
    def explode_depth_output(depth: Module.IStereoDepth.Output) -> list[Module.IStereoDepth.Output]:
        N = depth.depth.size(0)
        depths = depth.depth.split(split_size=1, dim=0)
        covs   = [None] * N if depth.cov  is None else depth.cov.split(split_size=1, dim=0)
        masks  = [None] * N if depth.mask is None else depth.mask.split(split_size=1, dim=0)
        return [
            Module.IStereoDepth.Output(depth=depth, cov=cov, mask=mask)
            for depth, cov, mask in zip(depths, covs, masks)
        ]
    
    @staticmethod
    def explode_match_output(match: Module.IMatcher.Output) -> list[Module.IMatcher.Output]:
        N = match.flow.size(0)
        flows = match.flow.split(split_size=1, dim=0)
        covs   = [None] * N if match.cov  is None else match.cov.split(split_size=1, dim=0)
        masks  = [None] * N if match.mask is None else match.mask.split(split_size=1, dim=0)
        return [
            Module.IMatcher.Output(flow=flow, cov=cov, mask=mask)
            for flow, cov, mask in zip(flows, covs, masks)
        ]
    
    @staticmethod
    def explode_stereo_data(data: StereoData) -> list[StereoData]:
        return [
            StereoData(
                T_BS=T_BS,
                K   =K,
                baseline=[baseline],
                height=data.height,
                width=data.width,
                time_ns=[time_ns],
                imageL=imageL,
                imageR=imageR
            )
            for (T_BS, K, baseline, time_ns, imageL, imageR) in zip(
                data.T_BS.split(split_size=1), data.K.split(split_size=1),
                data.baseline, data.time_ns,
                data.imageL.split(split_size=1), data.imageR.split(split_size=1)
            )
        ]
    
    @staticmethod
    def explode_stereo(frame: StereoFrame) -> list[StereoFrame]:
        assert frame.gt_pose is not None
        return [
            StereoFrame(idx=[i], stereo=stereo, gt_pose=pp.SE3(gt_pose)) 
            for i, stereo, gt_pose in zip(frame.idx, MACVO_Online.explode_stereo_data(frame.stereo), frame.gt_pose.split(split_size=7))
        ]
    
    @staticmethod
    def explode_stereo_pair(frame: DataFramePair[StereoFrame]) -> list[DataFramePair[StereoFrame]]:
        return [DataFramePair[StereoFrame](
            idx=[idx], cur=cur_frame, nxt=nxt_frame
        ) for idx, cur_frame, nxt_frame in zip(
            frame.idx, MACVO_Online.explode_stereo(frame.cur), MACVO_Online.explode_stereo(frame.nxt)
        )]
    

    def estimate(self, frame: DataFramePair[StereoFrame], optimize_niter: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            depth0, _       = self.frontend.estimate(None            , frame.cur.stereo)
            depth1, match01 = self.frontend.estimate(frame.cur.stereo, frame.nxt.stereo)

        return self.estimate_sample(frame, depth0, depth1, match01, optimize_niter)
        # results: list[None | torch.Tensor] = [None, None, None]
        # for frame_s, depth0_s, depth1_s, match01_s in zip(
        #     self.explode_stereo_pair(frame), 
        #     self.explode_depth_output(depth0), 
        #     self.explode_depth_output(depth1), 
        #     self.explode_match_output(match01)
        # ):
        #     # estimate_sample receives un-batched data samples.
        #     rpe, rte, roe = self.estimate_sample(frame_s, depth0_s, depth1_s, match01_s, optimize_niter)
        #     results[0] = results[0] + rpe if results[0] else rpe
        #     results[1] = results[1] + rte if results[1] else rte
        #     results[2] = results[2] + roe if results[2] else roe
        
        # assert results[0] and results[1] and results[2]
        # return (results[0], results[1], results[2])
