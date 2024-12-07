import torch
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader, ConcatDataset

from DataLoader import TrainDataset, DataFramePair, StereoFrame, IDataTransform
from Utility.Config import load_config
from Utility.PrettyPrint import Logger, ColoredTqdm
from Utility.Sandbox import Sandbox

from .System2 import MACVO_Online
from .Utils  import CircularBuffer, TrainInstabilityException, DataLogService, ScalarScheduler


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    logger = DataLogService("wandb", "MAC-Mod Learner bf16")
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
    
    rpe_history     = CircularBuffer[float](cfg.Trainer.log_freq)
    rte_history     = CircularBuffer[float](cfg.Trainer.log_freq)
    roe_history     = CircularBuffer[float](cfg.Trainer.log_freq)
    loss_history    = CircularBuffer[float](cfg.Trainer.log_freq)
    n_iter_schedule = ScalarScheduler(cfg.Trainer.niter_schedule.opt_steps, cfg.Trainer.niter_schedule.opt_niter)
    
    system  = MACVO_Online.from_config(cfg.System)
    optim_1 = torch.optim.AdamW(system.covmodel.modifier.parameters(), lr=cfg.Trainer.lr / cfg.Trainer.batch_size, weight_decay=cfg.Trainer.weight_decay)
    optim_1.zero_grad()
    
    step = 0
    epoch = 0
    while True:
        for idx, frame in enumerate(ColoredTqdm(trainloader)):
            try:
                rpe, rte, roe, = system.estimate(frame, optimize_niter=n_iter_schedule.get_value())
            except TrainInstabilityException as e:
                Logger.write("warn", f"TrainInstabilityException({e.msg}) Raised, skip current data point.")
                optim_1.zero_grad()
                continue
                
            step += 1
            
            loss = rte + roe
            loss.backward()
            
            loss_history.push(loss.cpu().detach().item())
            rpe_history.push(rpe.cpu().detach().item())
            rte_history.push(rte.cpu().detach().item())
            roe_history.push(roe.cpu().detach().item())
            
            if step % cfg.Trainer.batch_size == 0:
                n_iter_schedule.step()
                optim_1.step()
                optim_1.zero_grad()
                for g in optim_1.param_groups:
                    g['lr'] = cfg.Trainer.lr / (n_iter_schedule.get_value() * cfg.Trainer.batch_size)
            
            if step % cfg.Trainer.log_freq == 0:
                logger.write_entries(**{
                        "Loss"      : mean(loss_history.vec), 
                        "RPE"       : mean(rpe_history.vec),
                        "RTE"       : mean(rte_history.vec),
                        "ROE"       : mean(roe_history.vec),
                        "num_iter"  : n_iter_schedule.get_value(),
                        "epoch"     : epoch
                })
                Logger.write("info", f"{idx}/{len(trainloader)}, Loss={mean(loss_history.vec):10.5e}, RPE={mean(rpe_history.vec):10.5e}")
            
            if (step % cfg.Trainer.save_freq) == 0:
                torch.save(system.covmodel.modifier.state_dict(), result.path(f"modifier_{step}.pth"))
                Logger.write("info", f"Save to modifier_{step}.pth")
        
        epoch += 1
        Logger.write("warn", f"Finish an epoch")
