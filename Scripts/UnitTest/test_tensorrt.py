import pytest
import torch
from pathlib import Path


class MinimumNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_features=2, out_features=2)
    
    def forward(self, x: torch.Tensor):
        return self.linear(x)


@pytest.mark.trt
@pytest.mark.local
def test_trt_sanity():
    """
    This test simply tests if your environment supports TensorRT.
    It creates a minimum neural net and compiles it using TRT. If everything
    works, a TRT engine should be produced and simple inference can be run.
    """
    import tensorrt as trt
    from Utility.Extensions.TensorRTModule import TensorRTModule
    
    def fill_optimization_profile(builder: trt.Builder) -> tuple[trt.IOptimizationProfile, ...]:
        static_profile = builder.create_optimization_profile()
        static_shape   = trt.Dims([1, 256, 2])
        static_profile.set_shape("x", min=static_shape, opt=static_shape, max=static_shape)
        return (static_profile,)

    net = MinimumNetwork().cuda()
    mod = TensorRTModule.compile_torch_module(
        net, cache_to=Path("./cache/test"),
        input_names=("x",),
        output_names=("y",),
        inputs=(torch.zeros((1, 256, 2)).cuda(),),
        predict_output_shapes=lambda f: {"y": f["x"]},
        create_optimization_profiles=fill_optimization_profile,
        
        use_cache=False
    )
    assert mod is not None, "The TensorRT on your platform is not configured correctly."
    
    x = torch.zeros((1, 256, 2), device='cuda')
    y = mod(x)["y"]
    
    assert y.shape == x.shape, "The result did not pass sanity check. "\
        "I don't know why, but there are definitely some problem in environment configuration."
