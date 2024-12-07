from types import SimpleNamespace
import torch

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import StereoFrame
from Module.Map import BatchFrame, BatchObservation, BatchPoints, TensorMap
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Visualizer import PLTVisualizer
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestable


from .Interface import IVisualOdometry

class MACVO(IVisualOdometry[StereoFrame], ConfigTestable):
    # Type alias of callback hooks for MAC-VO system. Will be called by the system on
    # certain event occurs (optimization finish, for instance.)
    T_SYSHOOK = Callable[["MACVO",], None]
    
    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile,
        frontend        : Module.IFrontend, 
        motion_model    : Module.IMotionModel,
        kp_selector     : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.IObservationCov,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector,
        optimizer       : Module.IOptimizer,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")
        
        self.gmap = TensorMap()
        self.device = device
        self.match_cov_default = match_cov_default

        # Modules
        self.Frontend = frontend
        self.MotionEstimator = motion_model
        self.KeypointSelector = kp_selector
        self.OutlierFilter = obs_filter
        self.ObsCovModel = obs_covmodel
        self.MapRefiner = post_process
        self.KeyframeSelector = kf_selector
        self.Optimizer = optimizer
        # end

        self.min_num_point = 10
        self.num_point = num_point
        self.edge_width = edgewidth
        
        # Context
        self.prev_frame: StereoFrame | None = None
        self.prev_handle: int | None = None
        self.prev_depth_est: Module.IStereoDepth.Output | None = None
        
        # Hooks
        self.on_optimize_writeback: list[MACVO.T_SYSHOOK] = []

        self.report_config()
    
    @classmethod
    def from_config(cls: type["MACVO"], cfg: SimpleNamespace) -> "MACVO":
        odomcfg = cfg.Odometry
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        MotionEstimator     = Module.IMotionModel[StereoFrame].instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        ObservationFilter   = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObserveCovModel     = Module.IObservationCov.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner          = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector    = Module.IKeyframeSelector.instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)
        
        return cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            **vars(odomcfg.args),
        )
    
    def report_config(self):
        # Cute fine-print boxes
        box1 = Panel.fit(
            "\n".join(
                [
                    f"DepthEstimator cov: {self.Frontend.provide_cov[0]}",
                    f"MatchEstimator cov: {self.Frontend.provide_cov[1]}",
                    f"Observation cov:    {self.ObsCovModel.__class__.__name__}",
                ]
            ),
            title="Odometry Covariance",
            title_align="left",
        )
        box2 = Panel.fit(
            "\n".join(
                [
                    f"Frontend        -'{self.Frontend        .__class__.__name__}'",
                    f"MotionEstimator -'{self.MotionEstimator .__class__.__name__}'",
                    f"KeypointSelector-'{self.KeypointSelector.__class__.__name__}'",
                    f"OutlierFilter   -'{self.OutlierFilter   .__class__.__name__}'",
                    f"MapRefiner      -'{self.MapRefiner      .__class__.__name__}'",
                ]
            ),
            title="Odometry Modules",
            title_align="left",
        )
        GlobalConsole.print(Columns([box1, box2]))

    def new_keypoint(
        self, 
        data_frame: StereoFrame, 
        map_frame: BatchFrame,
        depth_est: Module.IStereoDepth.Output,
        match_est: Module.IMatcher.Output
    ) -> BatchObservation:
        """
        Generate new keypoint and register to TensorMap.

        Note
        ---
        Requires `self.DepthEstimator` to be initialized correctly. Will not call
        `self.DepthEstimator.receive_stereo(curr_frame)` internally.
        """
        assert len(map_frame) == 1
        assert map_frame.frame_idx is not None

        kp_uv = self.KeypointSelector.select_point(data_frame.stereo, self.num_point, depth_est, match_est)
        num_kp = kp_uv.size(0)
        
        kp_d     = self.Frontend.retrieve_pixels(kp_uv, depth_est.depth).squeeze(0)
        kp_d_cov = self.Frontend.retrieve_pixels(kp_uv, depth_est.cov)
        kp_d_cov = kp_d_cov.squeeze(0) if kp_d_cov is not None else None
        
        if kp_uv.size(0) == 0:
            # NOTE: Refer to https://github.com/pypose/pypose/issues/342
            kp_3d = torch.empty((0, 3))
        else:
            kp_3d = map_frame.pose[0].Act(pixel2point_NED(kp_uv, kp_d, data_frame.stereo.frame_K).cpu())   #type: ignore
        
        kp_covTc = self.ObsCovModel.estimate(data_frame.stereo, kp_uv, depth_est, None, kp_d_cov, flow_cov=None)

        # Record color of keypoints for visualization
        kp_uv_cpu = kp_uv.cpu()
        kp_color = data_frame.stereo.imageL[..., kp_uv_cpu[..., 1], kp_uv_cpu[..., 0]].squeeze(0).permute(1, 0)
        kp_color = (kp_color * 255).to(torch.uint8)
        
        # Convert covariance to world coordinate
        est_R = map_frame.pose.rotation().matrix().repeat((num_kp, 1, 1)).double()
        kp_covTw = torch.bmm(torch.bmm(est_R, kp_covTc), est_R.transpose(1, 2))
        
        # Register points and observations
        kp_idx = self.gmap.points.push(BatchPoints(kp_3d, kp_covTw, kp_color))
        fake_cov_uv = torch.ones((num_kp, 3)) * self.match_cov_default
        fake_cov_uv[..., 2] = 0.
        
        obs = BatchObservation(
            point_idx=kp_idx,
            frame_idx=map_frame.frame_idx[0].repeat((num_kp,)),
            pixel_uv=kp_uv_cpu, pixel_d=kp_d, cov_Tc=kp_covTc,
            cov_pixel_uv=fake_cov_uv,
            cov_pixel_d=(torch.ones((num_kp,)) * -1) if (kp_d_cov is None) else kp_d_cov
        )
        obs = obs[self.OutlierFilter.filter(obs)]
        self.gmap.add_observation(obs)
        
        return obs

    def match_keypoint(self,
                       to_frame_data: StereoFrame,
                       orig_obs: BatchObservation,
                       to_frame: BatchFrame,
                       depth_est : Module.IStereoDepth.Output,
                       match_est : Module.IMatcher.Output):
        """
        Given a set of observations, match these observations to corresponding kps
        in other frame.

        Note
        ---
        Requires `self.DepthEstimator` and `self.OutlierFilter` 
        to be initialized properly.
        `self.DepthEstimator` receives the SourceDataFrame of latest frame
        """
        assert len(to_frame) == 1
        assert to_frame.frame_idx is not None

        kp2_uv_old  = orig_obs.pixel_uv.to(match_est.flow.device)
        kp2_uv      = kp2_uv_old + self.Frontend.retrieve_pixels(kp2_uv_old, match_est.flow).T
        kp2_uv_cov  = self.Frontend.retrieve_pixels(kp2_uv_old, match_est.cov)
        kp2_uv_cov  = kp2_uv_cov.T if kp2_uv_cov is not None else None
        
        inbound_mask = filterPointsInRange(
            kp2_uv,
            (self.edge_width, to_frame_data.stereo.width - self.edge_width), 
            (self.edge_width, to_frame_data.stereo.height - self.edge_width)
        )

        orig_obs    = orig_obs[inbound_mask.cpu()]
        kp2_uv      = kp2_uv[inbound_mask]
        kp2_uv_cov  = None if (kp2_uv_cov is None) else kp2_uv_cov[inbound_mask]
        
        num_points = len(orig_obs)
        kp2_d      = self.Frontend.retrieve_pixels(kp2_uv, depth_est.depth).squeeze(0)
        kp2_d_cov  = self.Frontend.retrieve_pixels(kp2_uv, depth_est.cov)
        kp2_d_cov  = kp2_d_cov.squeeze(0) if kp2_d_cov is not None else None

        obs_covTc = self.ObsCovModel.estimate(to_frame_data.stereo, kp2_uv, depth_est, match_est, kp2_d_cov, kp2_uv_cov)
        
        obs = BatchObservation(
            point_idx=orig_obs.point_idx,
            frame_idx=to_frame.frame_idx[0].repeat((kp2_uv.size(0),)),
            pixel_uv=kp2_uv,
            pixel_d=kp2_d,
            cov_Tc=obs_covTc,
            cov_pixel_uv=(torch.ones((num_points, 2)) * self.match_cov_default) if (kp2_uv_cov is None) else kp2_uv_cov,
            cov_pixel_d=(torch.ones((num_points,)) * -1) if (kp2_d_cov is None) else kp2_d_cov
        )
        obs = obs[self.OutlierFilter.filter(obs)]
        self.gmap.add_observation(obs)
        
        return obs
    
    def epilog(self, curr_frame: StereoFrame, depth_est: Module.IStereoDepth.Output) -> None:
        self.prev_frame = curr_frame
        self.prev_handle = len(self.gmap.frames) - 1
        self.prev_depth_est = depth_est
        
    def initialize(self, curr_frame: StereoFrame) -> None:
        # Estimate Depth
        depth, _ = self.Frontend.estimate(None, curr_frame.stereo)
        est_pose = self.MotionEstimator.predict(curr_frame, None, depth.depth)
        
        # num_obs is set to zero since this will be updated in new_keypoint later
        self.OutlierFilter.set_meta(curr_frame.stereo)
        self.gmap.add_frame(curr_frame.stereo.K, est_pose, 0, curr_frame.stereo.T_BS, None)
        self.MotionEstimator.update(self.gmap.frames[-1].squeeze().pose)
        
        # Trigger callback functions on optimization finish (though there is no optimization
        # happening) for the first frame.
        for func in self.on_optimize_writeback: func(self)
        
        # Store context
        self.epilog(curr_frame, depth)

    @Timer.cpu_timeit("Odom_Runtime")
    @Timer.gpu_timeit("Odom_Runtime")
    def run(self, frame: StereoFrame) -> None:
        if self.prev_frame is None:
            return self.initialize(frame)
        assert self.prev_handle is not None
        assert self.prev_depth_est is not None
        
        if not self.KeyframeSelector.isKeyframe(frame):
            self.gmap.add_frame(
                frame.stereo.K,
                self.gmap.frames[self.prev_handle].pose,
                0, 
                frame.stereo.T_BS,
                None, 
                flag=BatchFrame.FLAG_NEED_INTERP
            )
            return
    
        # Frontend inference
        depth_est, match_est = self.Frontend.estimate(self.prev_frame.stereo, frame.stereo)
        
        # NOTE: should always writeback optimized pose to global map before selecting new keypoints (register
        # new 3D point) on that frame.
        self.Optimizer.write_map(self.gmap)
        # Trigger callback functions (useful in online systems like ROS where 
        # we want to do something immediately after the optimization result is ready)
        for func in self.on_optimize_writeback: func(self)
        
        # Update motion model (this must be after write_back to get latest result)
        self.MotionEstimator.update(self.gmap.frames[self.prev_handle].squeeze().pose)
        
        # Make new prediction
        est_pose = self.MotionEstimator.predict(frame, match_est.flow, depth_est.depth)
        
        # We will use the estimated matching quality between frame (t-1) and frame t
        # to select new keypoints on frame (t-1).
        prev_obs = self.new_keypoint(self.prev_frame, self.gmap.frames[self.prev_handle], self.prev_depth_est, match_est)
        
        self.gmap.add_frame(frame.stereo.K, est_pose, 0, frame.stereo.T_BS, None)
        curr_frame_map = self.gmap.frames[-1]
        
        # Update keypoints from previous frame to current frame
        new_obs = self.match_keypoint(frame, prev_obs, curr_frame_map, depth_est, match_est)
        
        # Visualizer start
        PLTVisualizer.visualize_imatcher("Matching", match_est, self.prev_frame, frame)
        PLTVisualizer.visualize_Obs("observation", self.prev_frame.stereo.imageL, frame.stereo.imageL, new_obs, depth_est.cov, match_est.cov, None)
        # Visualizer end
        
        if len(new_obs) > 0:
            curr_frame_map.quality[0] = new_obs.cov_Tc.det().mean()
        else:
            curr_frame_map.quality[0] = -1
        
        if len(new_obs) < self.min_num_point:
            Logger.write("error", f"VOLostTrack @ {frame.frame_idx} - only get {len(new_obs)} observations")

            curr_frame_map.flag |= curr_frame_map.FLAG_VO_LOSTTRACK
            self.gmap.frames.update(curr_frame_map, self.gmap.frames.Scatter.FLAG | self.gmap.frames.Scatter.QUALITY)
            
            self.epilog(frame, depth_est)
            return
        
        # Construct graph optimization problem and execute optimization
        self.Optimizer.optimize(self.gmap, [-1])
        
        self.epilog(frame, depth_est)
    
    def terminate(self) -> None:
        super().terminate()
        self.Optimizer.write_map(self.gmap)
        self.Optimizer.terminate()
        
        self.gmap, _ = self.MapRefiner.elaborate_map(self.gmap)

    def get_map(self) -> TensorMap:
        return self.gmap
    
    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        Module.IKeyframeSelector.is_valid_config(config.keyframe)
        Module.IMapProcessor.is_valid_config(config.postprocess)
        Module.IObservationFilter.is_valid_config(config.outlier)
        Module.IMotionModel.is_valid_config(config.motion)
        Module.IKeypointSelector.is_valid_config(config.keypoint)
        Module.IObservationCov.is_valid_config(config.cov.obs)
        Module.IFrontend.is_valid_config(config.frontend)
        
        cls._enforce_config_spec(config.args, {
            "device"            : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "num_point"         : lambda b: isinstance(b, int) and b > 0, 
            "edgewidth"         : lambda b: isinstance(b, int) and b > 0, 
            "match_cov_default" : lambda b: isinstance(b, (float, int)) and b > 0.0, 
            "profile"           : lambda b: isinstance(b, bool),
        })

