import argparse
import torch
import rerun as rr
from pathlib import Path

from DataLoader import SequenceBase, StereoFrame, IDataTransform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO

from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table, ColoredTqdm
from Utility.Sandbox import Sandbox
from Utility.Visualizer import PLTVisualizer
from Utility.Visualizer import RerunVisualizer as rrv
from Utility.Timer import Timer



def VisualizeRerunCallback(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
    PLTVisualizer.visualize_stereo("stereo", frame.stereo.imageL, frame.stereo.imageR)
    rr.set_time_sequence("frame_idx", frame.frame_idx)
    
    if rrv.status == rrv.State.INACTIVE: return
    
    # Non-key frame does not need visualization
    if system.gmap.frames[-1].FLAG_NEED_INTERP & int(system.gmap.frames.flag[-1].item()) != 0: return 
    
    if frame.frame_idx > 0:
        rrv.visualizeTrajectory(system.gmap)

    rrv.visualizeFrameAt(system.gmap, -1)
    rrv.visualizeRecentPoints(system.gmap)
    rrv.visualizeImageOnCamera(frame.stereo.imageL[0].permute(1, 2, 0))

def VisualizeVRAMUsage(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_reserved(0) / 1e9  # Convert to GB
        allocated_memory = f"{round(allocated_memory, 3)} GB"
    else:
        allocated_memory = "N/A"
    
    pb.set_description(desc=f"{system.gmap}, VRAM={allocated_memory}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom", type=str, default = "Config/Experiment/MACVO/MACVO.yaml")
    parser.add_argument("--data", type=str, default = "Config/Sequence/TartanAir_seaside_000.yaml")
    parser.add_argument(
        "--seq_to",
        type=int,
        default=-1,
        help="Crop sequence to frame# when ran. Set to -1 (default) if wish to run whole sequence",
    )
    parser.add_argument(
        "--seq_from",
        type=int,
        default=0,
        help="Crop sequence from frame# when ran. Set to 0 (default) if wish to start from first frame",
    )
    parser.add_argument(
        "--resultRoot",
        type=str,
        default="./Results",
        help="Directory to store trajectory and files generated by the script."
    )
    parser.add_argument(
        "--useRR",
        action="store_true",
        help="Activate RerunVisualizer to generate <config.Project>.rrd file for visualization.",
    )
    parser.add_argument(
        "--saveplt",
        action="store_true",
        help="Activate PLTVisualizer to generate <frame_idx>.jpg file in space folder for covariance visualization.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload entire trajectory into RAM to reduce data fetching overhead during runtime."
    )
    parser.add_argument(
        "--autoremove",
        action="store_true",
        help="Cleanup result sandbox after script finishs / crashed. Helpful during testing & debugging."
    )
    parser.add_argument(
        "--noeval", 
        action="store_true",
        help="Evaluate sequence after running odometry."
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Record timing for system (active Utility.Timer for global time recording)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Metadata setup & visualizer setup
    cfg, cfg_dict = load_config(Path(args.odom))
    odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]
    datacfg, datacfg_dict = load_config(Path(args.data))
    project_name = odomcfg.name + "@" + datacfg.name

    exp_space = Sandbox.create(Path(args.resultRoot), project_name)
    if args.autoremove: exp_space.set_autoremove()
    exp_space.config = {
        "Project": project_name,
        "Odometry": odomcfg_dict,
        "Data": {"args": datacfg_dict, "end_idx": args.seq_to, "start_idx": args.seq_from},
    }

    # Setup logging and visualization
    rrv.setup(project_name, True, Path(exp_space.folder, "rrvis.rrd"), useRR=args.useRR)
    Timer.setup(active=args.timing)
    
    if args.saveplt:
        PLTVisualizer.setup(PLTVisualizer.State.SAVE_FILE, save_path=Path(exp_space.folder), dpi=400)

    def onFrameFinished(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
        VisualizeRerunCallback(frame, system, pb)
        VisualizeVRAMUsage(frame, system, pb)

    # Initialize data source
    sequence = SequenceBase[StereoFrame].instantiate(datacfg.type, datacfg.args)\
        .clip(args.seq_from, args.seq_to)\
        .transform([
            IDataTransform.instantiate(tcfg.type, tcfg.args)
            for tcfg in cfg.Preprocess
        ])
    
    if args.preload:
        sequence = sequence.preload()
    
    system = MACVO.from_config(asNamespace(exp_space.config))
    system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)
    
    rrv.visualizeTrajectory(system.get_map())
    rrv.visualizePointCov(system.get_map())
    
    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    if not args.noeval:
        header, result = EvaluateSequences([str(exp_space.folder)], correct_scale=False)
        print_as_table(header, result)
