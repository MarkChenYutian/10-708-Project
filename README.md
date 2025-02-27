# MAC-Mod: Matric-Aware Covariance Modifier for Stereo Visual Odometry

This is preliminary work on the learning-based covariance modifier for the 10-708 course project. The codebase is only published as per the course project requirement.

**Disclaimer**: The commit is squashed into a single commit for security reasons as we used proprietary CI/CD tools and the repository is linked with other internal projects. Therefore, the authorship & contribution of GitHub does not accurately reflect the work distribution between the team members.

**Disclaimer 2**: This is an internal work in progress, we will not provide any further update on this repository after Dec 6, 2024.

## Run Training Script

```bash
python -m Train.CovModifier.Train --config ./Config/Train/CovModifier/AsModifier.yaml
```

## Run Experiments

```bash
python -m Scripts.Experiment.Experiment_MACVO --config ./Config/Experiment/MACVO/Mod/MACVO_corrector2.yaml
```

## Run Evaluation and Plotting

```bash
  $ python -m Evaluation.EvalSeq --spaces SPACE_0, [SPACE, ...]
  $ python -m Evaluation.PlotSeq --spaces SPACE_0, [SPACE, ...]
```

---
Below is the README.md content from MAC-VO.

https://github.com/user-attachments/assets/f7f33f28-5de7-412b-8f60-b0fcab91d48e


> [!NOTE]  
> We plan to continue to develop and release updates to the MAC-VO system. This includes releasing TensorRT accelerated implementation, adapting more frontend networks, and integrating with ROS2. If you are interested, please consider star ⭐ this repo to stay tuned.

> [!NOTE]  
> Clone the repository using the following command to include all submodules automatically.
> ```bash
> git clone git@github.com:MAC-VO/MAC-VO.git --recursive
> ```


## Environment & Requirements

**Docker Image** See `/Docker/DockerfileRoot` to build the container.

**Virtual Environment**, Need Python 3.10+. See `requirements.txt` for environment requirements. Only tested on `PyTorch >=2.0` and `NumPy < 2.0.0`. *Breaking Change: must have PyPose >= 0.6.8*

*Optional* - To use TensorRT for best performance on inference, you need to install `tensorrt` package.


* All pretrained model for MACVO, stereo TartanVO and DPVO is in our [release](https://github.com/MAC-VO/MAC-VO/releases/tag/model) 

## Demo

### Run MAC-VO for experiment

* Download a demo sequence through [drive](https://drive.google.com/file/d/1kCTNMW2EnV42eH8g2STJHcVWEbVKbh_r/view?usp=sharing) and the pre-trained model for [front-end model](https://github.com/MAC-VO/MAC-VO/releases/download/Weight-Release/MACVO_FrontendCov.pth) and [posenet model](https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl)
* Remember to change the `DATAPATH` in the config file

* **Run MAC-VO on a single sequence**

  ```bash
  $ python MACVO.py --odom [PATH_TO_ODOM_CONFIG] --data [PATH_TO_DATA_CONFIG] --useRR
  ```

  see `python MACVO.py --help` for more flags and configurations.

  <details>
  <summary>
  Example Usages (See More)
  </summary>

  - Run MAC-VO (*Ours* method):
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
    ```

  - Run MAC-VO for ablation studies
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/Ablation_Study/[CHOOSE_ONE_CFG].yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml --useRR
    ```
  
  </details>

* **Run MAC-VO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_MACVO --odom [PATH_TO_ODOM_CONFIG]
  ```

* **Run MAC-VO Mapping Mode**

  Mapping mode only reprojects pixels to 3D space and does *not* optimize the pose. To run the mapping mode, you need to first run a trajectory through the original mode (MAC-VO), 
  and pass the resulting pose file to MAC-VO mapping mode by modifying the config. (Specifically, `motion > args > pose_file` in config file)

  ```bash
  $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO_MappingMode.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
  ```

### Analyze Results and Plotting

Every run will produce a `Sandbox` (or `Space`). A `Sandbox` is a storage unit that contains all the results and meta-information of an experiment. The evaluation and plotting script usually requires one or more paths of sandbox(es).

* **Evaluate Sequence(s)**

  Calculate the absolute translate error (ATE, m); relative translation error (RTE, m/frame); relative orientation error (ROE, deg/frame); relative pose error (per frame on se(3)).

  ```bash
  $ python -m Evaluation.EvalSeq --spaces SPACE_0, [SPACE, ...]
  ```

* **Plot Sequence(s)**

  Plot sequences, translation, translation error, rotation and rotation error.

  ```bash
  $ python -m Evaluation.PlotSeq --spaces SPACE_0, [SPACE, ...]
  ```

<details>
<summary>
See more commands for plotting figures / creating demo.
</summary>

We used [the Rerun](https://rerun.io) visualizer to visualize 3D space including camera pose, point cloud and trajectory.

* **Create Rerun Recording for Run(s)**

  ```bash
  $ python -m Scripts.AdHoc.DemoCompare --macvo_space [MACVO_RESULT_PATH] --other_spaces [RESULT_PATH, ...] --other_types [{DROID-SLAM, DPVO, TartanVO}, ...]
  ```

* **Create Rerun Visualization for Map**

  Create a `tensor_map_vis.rrd` file in each sandbox that stores the visualization of 3D point cloud map.

  ```bash
  $ python -m Scripts.AdHoc.DemoCompare --spaces [RESULT_PATH, ...] --recursive?
  ```

* **Create Rerun Visualization for a Single Run** (Eye-catcher figure for our paper)

  ```bash
  $ python -m Scripts.AdHoc.DemoSequence --space [RESULT_PATH] --data [DATA_CONFIG_PATH]
  ```

</details>

### Run Baseline Experiments

We also integrated two baseline methods (DPVO, TartanVO Stereo) into the codebase for evaluation, visualization and comparison.

<details>
<summary>
Expand All (4 commands)
</summary>

* **Run DPVO on a single sequence**

  ```bash
  $ python DPVO.py --odom ./Config/Experiment/Baseline/DPVO/DPVO.yaml --data [PATH_TO_DATA_CONFIG]
  ```

* **Run DPVO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_DPVO --odom ./Config/Experiment/Baseline/DPVO/DPVO.yaml
  ```

* **Run TartanVO (Stereo) on a single sequence**

  ```bash
  $ python TartanVO.py --odom ./Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml --data [PATH_TO_DATA_CONFIG]
  ```

* **Run TartanVO (Stereo) on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_TartanVO --odom ./Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml
  ```

</details>


## Coordinate System in this Project

* PyTorch Tensor Data - All images are stored in `BxCxHxW` format following the convention. Batch dimension is always the first dimension of tensor.

* Pixels on Camera Plane - All pixel coordinates are stored in `uv` format following the OpenCV convention, where the direction of uv are "east-down". **Note that this requires us to access PyTorch tensor in `data[..., v, u]`** indexing.
  
  ```
  (0, 0)----> u
    |
    |
    v
  ```
* World Coordinate - `NED` convention, `+x -> North`, `+y -> East`, `+z -> Down` with the first frame being world origin having identity SE3 pose.


## Customization, Extension and Future Developement

> This codebase is designed with *modularization* in mind so it's easy to modify, replace, and re-configure modules of MAC-VO. One can easily use or replase the provided modules like flow estimator, depth estimator, keypoint selector, etc. to create a new visual odometry.

🤗 We welcome everyone to extend and redevelop the MAC-VO. For documentation please visit the [Documentation Site](https://mac-vo.github.io/wiki/)
