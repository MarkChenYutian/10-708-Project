import torch
from abc import abstractmethod
from types import SimpleNamespace
from dataclasses import dataclass

from Utility.Extensions import ConfigTestableSubclass


class ICovarianceModifier(torch.nn.Module, ConfigTestableSubclass):
    """
    Taking in some information and the initial estimation of covariance blocks (3x3) of keypoint, 
    estimate an covariance update term delta_cov to be applied on the estimation for more accurate result.
    
    delta <- delta + ICovarianceModifier(Input)
    """
    
    @dataclass
    class Input:
        K         : torch.Tensor    # torch.float32 of shape 3x3       | Camera intrinsic
        keypoint  : torch.Tensor    # torch.float32 of shape Nx2       | 2D keypoint uv coordinate
        image     : torch.Tensor    # torch.float32 of shape 1x3xHxW   | 
        depth     : torch.Tensor    # torch.float32 of shape 1x1xHxW   | 
        
        flow      : torch.Tensor | None = None # torch.float32 of shape 1x2xHxW      | Flow map
        cov_blocks: torch.Tensor | None = None # torch.float64 of shape Nx3x3        | 3D covariance block estimated by some other model
        depth_cov : torch.Tensor | None = None # torch.float32 of shape Nx1xHxW      | Local depth covariance patch
        flow_cov  : torch.Tensor | None = None # torch.float32 of shape Nx3xHxW      | Flow covariance of specific keypoint
    
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
    
    @staticmethod
    def _extract_patches(values: list[torch.Tensor] | torch.Tensor, keypoint: torch.Tensor, patch_size: int) -> list[torch.Tensor]:
        """
        Given a list of torch.Tensor with shape (..., H, W) and N keypoints of uv coordinate with a patch size P,
        Returns a list of patches (torch.Tensor) with shape (..., N, P, P)
        """
        assert patch_size % 2 == 1
        half_patch = patch_size // 2
        if not isinstance(values, list): values = [values]
        
        keypoint_long = keypoint.long().to(values[0].device)
        kp_u, kp_v = keypoint_long[..., 0], keypoint_long[..., 1]

        # Get local depth average and variance
        u_indices = torch.arange(-half_patch, half_patch + 1, dtype=torch.long, device=values[0].device)
        v_indices = torch.arange(-half_patch, half_patch + 1, dtype=torch.long, device=values[0].device)
        uu, vv = torch.meshgrid(u_indices, v_indices, indexing="ij")

        all_u_indices = kp_u.unsqueeze(-1) + uu.reshape(1, -1)
        all_v_indices = kp_v.unsqueeze(-1) + vv.reshape(1, -1)
        
        results = [value[..., all_v_indices, all_u_indices] for value in values]
        shapes  = [[r.size(d) for d in range(r.dim() - 1)] + [patch_size, patch_size] for r in results]
        
        return [r.reshape(shape) for r, shape in zip(results, shapes)]


    @abstractmethod
    def forward(self, x: Input) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Given some input (defined above), output Nx3x3 torch.Tensor with dtype=torch.float64
        The output will be added directly on the original covariance. Note that the result must
        be a valid covariance matrix! (positive semi-definite and symmetric)
        """
        ...
