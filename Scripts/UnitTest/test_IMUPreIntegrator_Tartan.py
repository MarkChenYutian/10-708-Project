import argparse
from pathlib import Path

import pypose as pp
import rerun as rr
import torch
from Utility.Config import load_config
from pypose.module import IMUPreintegrator

from DataLoader import TartanAir_Sequence

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ips", type=int, default=100)
    args.add_argument("--useRR", action="store_true")
    args = args.parse_args()
    useRR: bool = args.useRR

    datacfg, datacfg_dict = load_config(Path("Config/Sequence/TartanAir_abandonfac_001.yaml"))
    dataroot = datacfg_dict["root"]

    state_pts = torch.zeros((0, 3))
    imugt_pts = torch.zeros((0, 3))
    if useRR:
        rr.init("IMUPreIntegrator", spawn=True)
        # rr.connect()
        # rr.save("./IMUPreIntegrator_visualize.rrd")

    Sequence = TartanAir_Sequence({
        "root": dataroot,
        "compressed": True,
        "gtFlow": False,
        "gtDepth": False,
        "gtPose": False,
        "imu_freq": args.ips,
        "imu_sim": TartanAir_Sequence.default_imu_simulate_spec,
    })
    
    frame = Sequence[0]
    assert (frame.imu.init_pos is not None) and (frame.imu.init_rot is not None) and (frame.imu.init_vel is not None)
    Integrator = IMUPreintegrator(
        frame.imu.init_pos,
        frame.imu.init_rot,
        frame.imu.init_vel,
        gravity=frame.imu.frame_gravity,
        reset=False,
    )

    if useRR:
        assert (frame.imu.gt_pos is not None) and (frame.imu.gt_rot is not None)
        state_pts = torch.cat([state_pts, frame.imu.init_pos[-1:]], dim=0)
        imugt_pts = torch.cat([imugt_pts, frame.imu.gt_pos[-1:]], dim=0)
        rr.log(
            "TransError_m",
            rr.Scalar(
                ((frame.imu.init_pos[-1] - frame.imu.gt_pos[-1]).norm()).item()
            )
        )
        rr.log(
            "RotationGT_0",
            rr.Scalar(pp.euler(frame.imu.gt_rot[-1])[0].item())
        )
        rr.log(
            "RotationGyro_0",
            rr.Scalar(
                pp.euler(frame.imu.init_rot[-1])[0].item()
            ),
        )
        rr.log(
            "RotationGT_1",
            rr.Scalar(pp.euler(frame.imu.gt_rot[-1])[1].item())
        )
        rr.log(
            "RotationGyro_1",
            rr.Scalar(
                pp.euler(frame.imu.init_rot[-1])[1].item()
            ),
        )
        rr.log(
            "RotationGT_2",
            rr.Scalar(pp.euler(frame.imu.gt_rot[-1])[2].item())
        )
        rr.log(
            "RotationGyro_2",
            rr.Scalar(
                pp.euler(frame.imu.init_rot[-1])[2].item()
            ),
        )
        rr.log(
            "RotationError_rad",
            rr.Scalar(
                (
                    (frame.imu.init_rot[-1].Inv() @ frame.imu.gt_rot[-1]).Log().norm() # type: ignore
                ).item()
            ),
        )
    
    init_flag = True
    for frame in Sequence:
        if init_flag:
            init_flag = False
            continue

        imu = frame.imu
        assert imu is not None
        state = Integrator(dt=imu.time_delta.double() / 1000, gyro=imu.gyro, acc=imu.acc, rot=imu.gt_rot)

        if useRR:
            assert (imu.gt_pos is not None) and (imu.gt_rot is not None)
            state_pts = torch.cat([state_pts, state["pos"][0, -1:]], dim=0)
            imugt_pts = torch.cat([imugt_pts, imu.gt_pos[-1:]], dim=0)

            rr.log(
                "RotationGT_0",
                rr.Scalar(pp.euler(imu.gt_rot[-1])[0].item())
            )
            rr.log(
                "RotationGyro_0",
                rr.Scalar(
                    pp.euler(state["rot"])[0,-1][0].item()
                ),
            )
            rr.log(
                "RotationGT_1",
                rr.Scalar(pp.euler(imu.gt_rot[-1])[1].item())
            )
            rr.log(
                "RotationGyro_1",
                rr.Scalar(
                    pp.euler(state["rot"])[0,-1][1].item()
                ),
            )
            rr.log(
                "RotationGT_2",
                rr.Scalar(pp.euler(imu.gt_rot[-1])[2].item())
            )
            rr.log(
                "RotationGyro_2",
                rr.Scalar(
                    pp.euler(state["rot"])[0,-1][2].item()
                ),
            )
            rr.log(
                "RotationError_rad",
                rr.Scalar(
                    (
                        (state["rot"][0, -1].Inv() @ imu.gt_rot[-1]).Log().norm() # type: ignore
                    ).item()
                ),
            )
            rr.log(
                "TransError_m",
                rr.Scalar(
                    ((state["pos"][0, -1] - imu.gt_pos[-1]).norm()).item()
                ),
            )

            rr.log("StatePoints", rr.Points3D(state_pts))
            rr.log("IMUGTPoints", rr.Points3D(imugt_pts))
