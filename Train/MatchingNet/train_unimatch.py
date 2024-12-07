import argparse
import os
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn

from typing import get_args
from types import SimpleNamespace as NS
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType, AddImageNoise, ScaleFrame
from Train.MatchingNet.loss import sequence_loss
from Utility.Config import load_config
from Utility.PrettyPrint import ColoredTqdm

from .utils import T_TrainType, AssertLiteralType, get_scheduler, get_optimizer



def train(modelcfg, cfg, loader):
    from Module.Network.UniMatchCov import UniMatchCov
    train_mode: T_TrainType = modelcfg.training_mode
    AssertLiteralType(train_mode, T_TrainType)
    
    # Trainig from a UniMatch checkpoint (no cov weight provided)
    model = UniMatchCov(fwd_kwargs=vars(modelcfg.fwd_args), **vars(modelcfg.args))
    if hasattr(modelcfg, "restore_ckpt"):
        model.load_unimatch_ckpt(modelcfg.restore_ckpt)
    else:
        model.load_ckpt(modelcfg.cov_ckpt)
    
    model.cuda()
    model.train()
    
    optimizer = get_optimizer(cfg.Model.optimizer.type)(
        model.parameters(),
        **vars(cfg.Model.optimizer.args)
    )
    scheduler = get_scheduler(cfg.Model.scheduler.type)(
        optimizer,
        **vars(cfg.Model.scheduler.args)
    )
    scaler = GradScaler(enabled=modelcfg.mixed_precision)
    model_ptr: UniMatchCov = model.module if isinstance(model, nn.DataParallel) else model
    
    match train_mode:
        case "flow":
            for param in model_ptr.cov_module.parameters():
                param.requires_grad = False
                
        case "cov":
            for param in model_ptr.unimatch.parameters():
                param.requires_grad = False
            for param in model_ptr.cov_module.parameters():
                param.requires_grad = True
            for param in model_ptr.upsampler.parameters():
                param.requires_grad = True
            if model_ptr.transformer is not None:
                for param in model_ptr.transformer.parameters():
                    param.requires_grad = True
        
        case "finalcov":
            for param in model_ptr.unimatch.parameters():
                param.requires_grad = False
            for param in model_ptr.cov_module.parameters():
                param.requires_grad = True
            for param in model_ptr.upsampler.parameters():
                param.requires_grad = True
            if model_ptr.transformer is not None:
                for param in model_ptr.transformer.parameters():
                    param.requires_grad = True
        case _:
            raise ValueError(f"Unavailable training mode {train_mode}")

    total_steps = 0
    should_keep_training = True
    while should_keep_training:
        frameData: DataFramePair[StereoFrame]
        loss     : torch.Tensor | None = None
        pb = ColoredTqdm(loader)
        for frameData in pb:
            metric_list = []
            
            assert frameData.cur.stereo.gt_flow   is not None
            assert frameData.cur.stereo.flow_mask is not None
            
            optimizer.zero_grad()
            img1, img2 = frameData.cur.stereo.imageL.cuda(), frameData.nxt.stereo.imageL.cuda()
            gt_flow = frameData.cur.stereo.gt_flow.cuda()
            flow_mask = frameData.cur.stereo.flow_mask.cuda()
            
            result = model(img1, img2)
            flow, cov = result["flow_preds"], result["flow_cov_preds"]
            loss, metrics = sequence_loss(
                                cfg=modelcfg,
                                preds=flow,
                                gt=gt_flow,
                                flow_mask=flow_mask,
                                cov_preds=cov)
            assert loss is not None

            metric_list.append(metrics)
            metrics["loss"] = loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), modelcfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metrics["lr"] = lr
            scaler.update()
            pb.set_description(desc=f"Loss={loss.item()}, Error={metrics['error']}, Cov={metrics['cov']}, cov_ratioe={metrics['cov_ratioe']}")
            if total_steps % int(modelcfg.log_freq) == 0:
                # print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                if modelcfg.wandb:
                    wandb.log(merge_matrices(metric_list))  #type: ignore
                metric_list = []
                
            total_steps += 1

            if total_steps > modelcfg.num_steps:
                should_keep_training = False
                break

            if modelcfg.autosave_freq and total_steps % modelcfg.autosave_freq == 0:
                PATH = "%s/%d.pth" % (modelcfg.autosave_dir, total_steps)  
                
                if isinstance(model, nn.DataParallel):
                    # We don't want to have a layer of `module.` on all weights. Since we are definitely not
                    # using DDP during inference, I will just save the "real weights" of the model.
                    torch.save(model.module.state_dict(), PATH)
                else:
                    torch.save(model.state_dict(), PATH)
                
    PATH = "%s/%d.pth" % (modelcfg.autosave_dir, total_steps)
    torch.save(model.state_dict(), PATH)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/Train/UniMatchCov_sm.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--autosave_dir", type=str, default="Model")
    parser.add_argument("--training_mode", type=str, choices=get_args(T_TrainType),
                        default="finalcov", help=f"Training mode: {get_args(T_TrainType)}")
    parser.add_argument("--run_id", default=None, help="Resume training from the checkpoint.")
    
    args = parser.parse_args()
    cfg, _ = load_config(Path(args.config))
    modelcfg = cfg.Model
    datacfg = cfg.Train
    modelcfg.time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    modelcfg.wandb = args.wandb
    modelcfg.autosave_dir = Path(args.autosave_dir, modelcfg.name + modelcfg.time)
    modelcfg.training_mode = args.training_mode
    
    os.makedirs(modelcfg.autosave_dir, exist_ok=True)
    torch.manual_seed(modelcfg.seed)
    np.random.seed(modelcfg.seed)
    
    transforms = [CenterCropFrame(NS(width=640, height=480)),
                  CastDataType(NS(dtype=cfg.Model.datatype)),
                  AddImageNoise(NS(stdv=5.0)),
                  ScaleFrame(NS(scale_u=cfg.Model.image_scale, scale_v=cfg.Model.image_scale, interp='nearest'))]
    
    traindatasets = TrainDataset.mp_instantiation(datacfg.data, 0, -1, lambda cfg: cfg.args.type in {"TartanAirv2", "TartanAir"})
    trainloader = DataLoader(
        ConcatDataset([
            ds.transform_source(transforms)
            for ds in traindatasets
            if ds is not None
        ]),
        batch_size=modelcfg.batch_size,
        shuffle=True,
        collate_fn=DataFramePair.collate,
        drop_last=True,
        num_workers=4,
    )
    
    if args.wandb:
        try:
            import wandb
            wandb.init(project="UniMatchCov", name = modelcfg.name,  config=modelcfg, 
                       resume="allow", id=args.run_id)
        except ImportError:
            print("Wandb is not installed, disabling it.")
            modelcfg.wandb = False

    train(modelcfg, cfg, trainloader)
