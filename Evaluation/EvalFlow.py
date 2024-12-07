import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace

from DataLoader import SequenceBase, StereoFrame, StereoData, ScaleFrame, NoTransform
from Module.Frontend.Matching import IMatcher
from Utility.PrettyPrint import ColoredTqdm, Logger
from Utility.Math import MahalanobisDist
from Utility.Plot import plot_image, plot_flow, plot_scalarmap, plot_flow_calibcurve, plot_flow_performance
from Utility.Datatypes import FlowPerformance, FlowCovPerformance


def plot_flow_instance(frame0: StereoData, frame1: StereoData, flow: IMatcher.Output, plot_to: str):
    GRID_SHAPE = (2, 3)
    _ = plt.figure(figsize=(8, 8), dpi=300)
    
    ax = plt.subplot2grid(GRID_SHAPE, (0, 0), rowspan=1, colspan=1)
    ax.set_title(f"Frame 0", loc="left")
    plot_image(ax, frame0.imageL.cpu().permute(0, 2, 3, 1)[0])
    
    ax = plt.subplot2grid(GRID_SHAPE, (1, 0), rowspan=1, colspan=1)
    ax.set_title(f"Frame 1", loc="left")
    plot_image(ax, frame1.imageL.cpu().permute(0, 2, 3, 1)[0])
    
    ax = plt.subplot2grid(GRID_SHAPE, (0, 1), rowspan=1, colspan=1)
    ax.set_title(f"Predict flow (masked)", loc="left")
    masked_flow = flow.flow
    if flow.mask is not None:
        masked_flow[~flow.mask.repeat(1, 2, 1, 1)] = 0.
    plot_flow(ax, masked_flow[0].detach().cpu())
    
    ax = plt.subplot2grid(GRID_SHAPE, (1, 1), rowspan=1, colspan=1)
    ax.set_title(f"GT flow", loc="left")
    assert frame0.gt_flow is not None
    gt_flow = frame0.gt_flow.clone()
    
    if frame0.flow_mask is not None:
        gt_flow[~(frame0.flow_mask).repeat(1, 2, 1, 1)] = 0.
    plot_flow(ax, gt_flow[0].detach().cpu())
    
    ax = plt.subplot2grid(GRID_SHAPE, (0, 2), rowspan=1, colspan=1)
    ax.set_title(f"Masked epe", loc="left")
    error       = (flow.flow.cpu() - gt_flow).square_()
    epe         = torch.sum(error, dim=1, keepdim=True).sqrt()
    if flow.mask is not None:
        epe[~flow.mask] = float('nan')
    cax = plot_scalarmap(ax, epe[0, 0])
    plt.colorbar(cax, ax=ax, orientation="vertical", fraction=0.05)
    
    ax = plt.subplot2grid(GRID_SHAPE, (1, 2), rowspan=1, colspan=1)
    ax.set_title(f"GT flow (Predicted Mask)", loc="left")
    gt_flow = frame0.gt_flow
    assert gt_flow is not None
    gt_flow = gt_flow
    if flow.mask is not None:
        gt_flow[~flow.mask.repeat(1, 2, 1, 1)] = 0.
    plot_flow(ax, gt_flow[0].detach().cpu())
    
    plt.tight_layout()
    plt.savefig(str(plot_to))
    plt.close()


@torch.inference_mode()
def evaluate_flow(matcher: IMatcher, seq: SequenceBase[StereoFrame], max_flow: float, huge_epe_warn: float | None = None, use_gt_mask: bool=False) -> FlowPerformance:
    prev_frame: StereoFrame | None = None
    results   : list[FlowPerformance]  = []
    
    frame: StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluating FlowModel"):
        if prev_frame is None:
            prev_frame = frame
            continue
        assert prev_frame.stereo.gt_flow is not None, "To evaluate flow quality, must use sequence with ground truth flow."
        
        match_out   = matcher.estimate(prev_frame.stereo, frame.stereo)
        est_flow    = match_out.flow
        gt_flow     = prev_frame.stereo.gt_flow.to(est_flow.device)
        
        error       = (est_flow - gt_flow).square_()
        epe         = torch.sum(error, dim=1, keepdim=True).sqrt()
        
        if use_gt_mask:
            mask        = prev_frame.stereo.flow_mask
            assert mask is not None
        else:
            mask        = est_flow < max_flow
            mask        = torch.logical_and(mask[:, :1], mask[:, 1:])
            if match_out.mask is not None:
                mask &= match_out.mask
        
        results.append(FlowPerformance(
            masked_epe= epe[mask].float().nanmean().item(),
            epe       = epe.float().nanmean().item(),
            px1       = (epe[mask] < 1).float().nanmean().item(),
            px3       = (epe[mask] < 3).float().nanmean().item(),
            px5       = (epe[mask] < 5).float().nanmean().item()
        ))
        
        if huge_epe_warn is not None and results[-1].masked_epe > huge_epe_warn:
            Logger.write("warn", f"Flow {prev_frame.frame_idx}->{frame.frame_idx} huge masked epe (> {huge_epe_warn}): epe={results[-1].masked_epe}")
            plot_flow_instance(prev_frame.stereo, frame.stereo, match_out, f"Flow_{prev_frame.frame_idx}-{frame.frame_idx}.png")
        prev_frame = frame
    
    plot_flow_performance(results, f"Flow_error_distribution.png")
    
    return FlowPerformance.mean(results)


@torch.inference_mode()
def evaluate_flowcov(matcher: IMatcher, seq: SequenceBase[StereoFrame], max_flow: float, use_gt_mask: bool=False) -> FlowCovPerformance:
    assert matcher.provide_cov, f"Cannot evaluate covariance for {matcher} since no cov is provided by the module."
    
    prev_frame: StereoFrame | None = None
    results: list[FlowCovPerformance] = []
    
    frame: StereoFrame
    for frame in ColoredTqdm(seq, desc="Evaluate FlowCov"):
        if prev_frame is None:
            prev_frame = frame
            continue
        assert prev_frame.stereo.gt_flow is not None, "To evaluate flow quality, must use sequence with ground truth flow."
        
        est_out = matcher.estimate(prev_frame.stereo, frame.stereo)
        est_flow, est_cov = est_out.flow, est_out.cov
        assert est_cov is not None
        
        gt_flow           = prev_frame.stereo.gt_flow.to(est_flow.device)
        
        error       = est_flow - gt_flow
        error2      = error.square()
        
        if use_gt_mask:
            mask = prev_frame.stereo.flow_mask
        else:
            mask        = est_flow < max_flow
            mask        = torch.logical_and(mask[:, :1], mask[:, 1:])
            if est_out.mask is not None: mask &= est_out.mask
        
        B, H, W     = est_flow.size(0), est_cov.size(2), est_cov.size(3)
        est_cov_mat = est_cov.permute(0, 2, 3, 1).diag_embed()  # [B x H x W x 2 x 2]
        error_mat   = error.permute(0, 2, 3, 1)                 # [B x H x W x 2]
        
        est_cov_mat = est_cov_mat.flatten(end_dim=2)            # [B*H*W x 2 x 2]
        error_mat   = error_mat.flatten(end_dim=2)              # [B*H*W x 2]
        
        likelihood  = est_cov_mat.det().log().view(-1, 1, 1) + MahalanobisDist(error_mat, torch.zeros_like(error_mat), est_cov_mat).square()
        likelihood  = likelihood.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        
        if est_out.mask is not None:
            mask &= est_out.mask

        uncertainty = est_cov_mat.det()
        q25_mask    = (uncertainty < uncertainty.nanquantile(0.25)).resize_as_(likelihood)
        q50_mask    = (uncertainty < uncertainty.nanquantile(0.5)).resize_as_(likelihood)
        q75_mask    = (uncertainty < uncertainty.nanquantile(0.75)).resize_as_(likelihood)
        
        uncertainty_masks = [
            torch.logical_and(
                est_cov > FlowCovPerformance.uncertainty_bin[idx],
                est_cov < FlowCovPerformance.uncertainty_bin[idx + 1]
            )
            for idx in range(len(FlowCovPerformance.uncertainty_bin) - 1)
        ]
        
        results.append(FlowCovPerformance(
            masked_nll        = likelihood[mask].nanmean().item(),
            q25_nll           = likelihood[q25_mask].nanmean().item(),
            q50_nll           = likelihood[q50_mask].nanmean().item(),
            q75_nll           = likelihood[q75_mask].nanmean().item(),
            uncertainty_avg_u = [error2[:, 0][uncertainty_masks[idx][:, 0]].nanmean().item() for idx in range(len(uncertainty_masks))],
            uncertainty_avg_v = [error2[:, 1][uncertainty_masks[idx][:, 1]].nanmean().item() for idx in range(len(uncertainty_masks))],
        ))
        
        prev_frame = frame
    
    mean_cov = FlowCovPerformance.mean(results)
    plot_flow_calibcurve(mean_cov, "output_calib.png")
    return mean_cov


if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from pathlib import Path
    
    torch.set_float32_matmul_precision('medium')
    
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, nargs="+", default=[])
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--scale_image", type=float, default=1.)
    args.add_argument("--max_flow", type=float, default=128.)
    args.add_argument("--gt_mask", action="store_true")
    args = args.parse_args()
    
    match_cfg, _ = load_config(Path(args.config))
    matcher = IMatcher.instantiate(match_cfg.type, match_cfg.args)
    
    if args.scale_image != 1.0:
        scale = args.scale_image
        assert isinstance(scale, (int, float))
        transform_fn = ScaleFrame(SimpleNamespace(scale_u=scale, scale_v=scale, interpolate="nearest"))
    else:
        transform_fn = NoTransform(SimpleNamespace())
    
    for data_path in args.data:
        data_cfg, _ = load_config(Path(data_path))
        seq         = SequenceBase[StereoFrame].instantiate(**vars(data_cfg)).preload()
        seq         = seq.transform(transform_fn)
        
        print(evaluate_flow(matcher, seq, max_flow=args.max_flow, huge_epe_warn=None, use_gt_mask=args.gt_mask))
        
        if matcher.provide_cov: print(evaluate_flowcov(matcher, seq, max_flow=args.max_flow, use_gt_mask=args.gt_mask))
        else: print(f"Skipped cov evaluation since module does not estimate covariance.")
