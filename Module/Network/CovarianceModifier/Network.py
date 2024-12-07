import torch
from types import SimpleNamespace
from .Interface import ICovarianceModifier


class NaiveNetworkModifier(ICovarianceModifier):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.register_buffer("_tril_mask", torch.tril(torch.ones(3, 3)).bool())
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12 , out_channels=128, kernel_size=3, stride=2),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            torch.nn.GELU(),
            # torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            # torch.nn.GELU(),
        ).to(self.config.device)
        
        self.adaptor = torch.nn.Sequential(
            torch.nn.Linear(in_features=128 + 6, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=7)
        ).to(self.config.device)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device": lambda s: isinstance(s, str) and (s == "cpu" or "cuda" in s.lower()),
            "patch_size": lambda v: isinstance(v, int) and v % 2 == 1,
        })
    
    def forward(self, x: ICovarianceModifier.Input) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.depth_cov is not None
        
        N = x.keypoint.size(0)
        H, W = x.image.size(-2), x.image.size(-1)
        dummy_patch = torch.zeros((1, H, W), dtype=torch.float32, device=self.config.device)
        
        if x.cov_blocks is not None:
            cov_L, err_code = torch.linalg.cholesky_ex(x.cov_blocks.to(self.config.device))
            if not (err_code == 0).all(): raise ValueError(f"LAPACK returns non-zero error code for Cholesky decomposition. (Error code: {err_code.item()})")
        else:
            cov_L = torch.zeros((N, 3, 3), device=self.config.device)
        
        patches = self._extract_patches([
                x.image[0].to(self.config.device), 
                x.depth[0].to(self.config.device), 
                x.depth_cov[0].to(self.config.device),
                dummy_patch.repeat(2, 1, 1) if x.flow is None else x.flow[0].to(self.config.device),
                dummy_patch.repeat(3, 1, 1) if x.flow_cov is None else x.flow_cov[0].to(self.config.device),
            ], 
            x.keypoint, self.config.patch_size
        )
        
        patch_size = self.config.patch_size
        position_enc = x.keypoint.T.unsqueeze(-1).unsqueeze(-1).to(self.config.device).float().repeat(1, 1, patch_size, patch_size)
        i = torch.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size, device=self.config.device)
        j = torch.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size, device=self.config.device)
        indices = torch.stack(torch.meshgrid(i, j, indexing="ij"), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1).permute(3, 0, 1, 2)
        position_enc = (position_enc + indices - 320.) / 320.
        patches.append(position_enc.detach())
        
        cov_L_flatten = cov_L[:, self._tril_mask].float()           # Nx6 lower triangular only (the tril of L in Sigma = LL^T)        
        patches_input = torch.cat(patches, dim=0).permute(1, 0, 2, 3)
        
        # Actual network part
        feat_encode = self.encoder(patches_input)
        feat_cat    = torch.cat([feat_encode.reshape(N, -1), cov_L_flatten], dim=-1)
        output      = self.adaptor(feat_cat)
        # End
        
        delta_cov_L = torch.zeros((N, 3, 3), device=self.config.device)
        delta_cov_L[:, self._tril_mask] = output[..., 1:]
        delta_cov = torch.bmm(delta_cov_L, delta_cov_L.transpose(-1, -2))     # Cholesky decomposition (inverse)
        
        combine_factor = torch.nn.functional.sigmoid(output[..., 0].unsqueeze(-1).unsqueeze(-1)).cpu().double()
        
        delta_cov = delta_cov.cpu().double()
        if self.training: return delta_cov, combine_factor
        else: return delta_cov.detach(), combine_factor.detach()


class ResidualCNNModifier(ICovarianceModifier):
    class Residual(torch.nn.Module):
        def __init__(self, module: torch.nn.Module | list[torch.nn.Module]) -> None:
            super().__init__()
            if isinstance(module, list):
                self.module = torch.nn.Sequential(*module)
            else:
                self.module = module
        
        def forward(self, x):
            return x + self.module(x)
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.register_buffer("_tril_mask", torch.tril(torch.ones(3, 3)).bool())
        
        patch_size = self.config.patch_size
        i = torch.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size, device=self.config.device)
        j = torch.linspace(-(patch_size - 1) / 2., (patch_size - 1) / 2., patch_size, device=self.config.device)
        indices = torch.stack(torch.meshgrid(i, j, indexing="ij"), dim=-1).unsqueeze(0).permute(3, 0, 1, 2)
        self.register_buffer('indices', indices)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12 , out_channels=128, kernel_size=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            torch.nn.GELU(),
            self.Residual([
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                torch.nn.GELU()
            ]),
            self.Residual([
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
                torch.nn.GELU()
            ]),
        ).to(self.config.device)
        
        self.adaptor = torch.nn.Sequential(
            torch.nn.Linear(in_features=128 + 6, out_features=128),
            torch.nn.GELU(),
            self.Residual([
                torch.nn.Linear(in_features=128, out_features=128),
                torch.nn.GELU(),
            ]),
            self.Residual([
                torch.nn.Linear(in_features=128, out_features=128),
                torch.nn.GELU(),
            ]),
            torch.nn.Linear(in_features=128, out_features=7)
        ).to(self.config.device)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device": lambda s: isinstance(s, str) and (s == "cpu" or "cuda" in s.lower()),
            "patch_size": lambda v: isinstance(v, int) and v % 2 == 1,
        })
    
    def forward(self, x: ICovarianceModifier.Input) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.depth_cov is not None
        
        N = x.keypoint.size(0)
        if N == 0: return torch.zeros((0, 3, 3), device='cpu'), torch.zeros((0,1,1), device='cpu')
        
        H, W = x.image.size(-2), x.image.size(-1)
        dummy_patch = torch.zeros((1, H, W), dtype=torch.float32, device=self.config.device)
        
        if x.cov_blocks is not None:
            cov_L, err_code = torch.linalg.cholesky_ex(x.cov_blocks.to(self.config.device))
            if not (err_code == 0).all(): raise ValueError(f"LAPACK returns non-zero error code for Cholesky decomposition. (Error code: {err_code.item()})")
        else:
            cov_L = torch.zeros((N, 3, 3), device=self.config.device)
        
        patches = self._extract_patches([
                x.image[0].to(self.config.device), 
                x.depth[0].to(self.config.device), 
                x.depth_cov[0].to(self.config.device),
                dummy_patch.repeat(2, 1, 1) if x.flow is None else x.flow[0].to(self.config.device),
                dummy_patch.repeat(3, 1, 1) if x.flow_cov is None else x.flow_cov[0].to(self.config.device),
            ], 
            x.keypoint, self.config.patch_size
        )
        
        patch_size = self.config.patch_size
        position_enc = x.keypoint.T.unsqueeze(-1).unsqueeze(-1).to(self.config.device).float().repeat(1, 1, patch_size, patch_size)
        position_enc = (position_enc + self.indices.repeat(1, N, 1, 1))
        position_enc[0] = (position_enc[0] - x.K[0, 0, 2]) / x.K[0, 0, 0]
        position_enc[1] = (position_enc[1] - x.K[0, 1, 2]) / x.K[0, 1, 1]
        patches.append(position_enc.detach())
        
        cov_L_flatten = cov_L[:, self._tril_mask].float()           # Nx6 lower triangular only (the tril of L in Sigma = LL^T)        
        patches_input = torch.cat(patches, dim=0).permute(1, 0, 2, 3)
        
        # Actual network part
        feat_encode = self.encoder(patches_input)
        feat_cat    = torch.cat([feat_encode.reshape(N, -1), cov_L_flatten], dim=-1)
        output      = self.adaptor(feat_cat)
        # End
        
        delta_cov_L = torch.zeros((N, 3, 3), device=self.config.device)
        delta_cov_L[:, self._tril_mask] = output[..., 1:]
        delta_cov = torch.bmm(delta_cov_L, delta_cov_L.transpose(-1, -2))     # Cholesky decomposition (inverse)
        
        combine_factor = torch.nn.functional.sigmoid(output[..., 0].unsqueeze(-1).unsqueeze(-1)).cpu().double()
        
        delta_cov = delta_cov.cpu().double()
        if self.training: return delta_cov, combine_factor
        else: return delta_cov.detach(), combine_factor.detach()
