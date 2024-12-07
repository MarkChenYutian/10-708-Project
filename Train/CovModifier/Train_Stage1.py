import torch
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader, ConcatDataset

from DataLoader import TrainDataset, DataFramePair, StereoFrame, IDataTransform
from Utility.Config import load_config
from Utility.PrettyPrint import Logger, ColoredTqdm
from Utility.Sandbox import Sandbox

from .System1 import CovarianceAligner
from .Utils  import CircularBuffer, TrainInstabilityException, DataLogService


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    logger = DataLogService("stdout", "MAC-Mod Aligner")
    result = Sandbox.create(Path("./Results"), "CovLearner")
    
    cfg, _  = load_config(Path(args.config))
    
    # Load training data
    datacfg = cfg.Train
    train_datasets = TrainDataset.mp_instantiation(
        datacfg.data, 0, -1, lambda cfg: cfg.type in {"TartanAirv2_NoIMU", "TartanAir_NoIMU"}
    )
    train_preprocess = [IDataTransform.instantiate(cfg.type, cfg.args) for cfg in datacfg.preprocess]
    trainloader = DataLoader[DataFramePair[StereoFrame]](
        ConcatDataset([
            ds.transform_source(train_preprocess) 
            for ds in train_datasets 
            if ds is not None
        ]),
        batch_size=1, shuffle=True, collate_fn=DataFramePair.collate, drop_last=True, num_workers=1,
    )
    # End
    
    loss_history    = CircularBuffer[float](cfg.Trainer.log_freq)
    
    system  = CovarianceAligner.from_config(cfg.System)
    optim_1 = torch.optim.AdamW(system.lrn_covmodel.modifier.parameters(), lr=cfg.Trainer.lr / cfg.Trainer.batch_size, weight_decay=cfg.Trainer.weight_decay)
    optim_1.zero_grad()
    
    step = 0
    while True:
        epoch = 0
        for idx, frame in enumerate(ColoredTqdm(trainloader)):
            try:
                lrn_cov0, lrn_cov1, ref_cov0, ref_cov1 = system.estimate(frame)
            except TrainInstabilityException as e:
                Logger.write("warn", f"TrainInstabilityException({e.msg}) Raised, skip current data point.")
                optim_1.zero_grad()
                continue
            
            try:
                lrn_cov0_L = torch.linalg.cholesky(lrn_cov0)
                lrn_cov1_L = torch.linalg.cholesky(lrn_cov1)
                ref_cov0_L = torch.linalg.cholesky(ref_cov0)
                ref_cov1_L = torch.linalg.cholesky(ref_cov1)
            except Exception as e:
                Logger.write("warn", f"TrainInstabilityException({e}) Raised, skip current data point.")
                optim_1.zero_grad()
                continue
                
            step += 1
            
            loss = (lrn_cov0_L - ref_cov0_L).square().mean() + (lrn_cov1_L - ref_cov1_L).square().mean()
            loss.backward()
            loss_history.push(loss.cpu().detach().item())
            
            if step % cfg.Trainer.batch_size == 0:
                optim_1.step()
                optim_1.zero_grad()
                for g in optim_1.param_groups:
                    g['lr'] = cfg.Trainer.lr / cfg.Trainer.batch_size
            
            if step % cfg.Trainer.log_freq == 0:
                logger.write_entries(**{
                        "Loss"      : mean(loss_history.vec), 
                        "epoch"     : epoch
                })
                Logger.write("info", f"{idx}/{len(trainloader)}, Loss={mean(loss_history.vec):10.5e}")
            
            if (step % cfg.Trainer.save_freq) == 0:
                torch.save(system.lrn_covmodel.modifier.state_dict(), result.path(f"modifier_{step}.pth"))
                Logger.write("info", f"Save to {result.path(f'modifier_{step}.pth')}")
        
        epoch += 1
        Logger.write("warn", f"Finish an epoch")
