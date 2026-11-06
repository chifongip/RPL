# **[CoRL 2026] RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains**

<p align="center">
  Yuanhang Zhang, Younggyo Seo, Juyue Chen, Yifu Yuan, Koushil Sreenath,
  Pieter Abbeel<sup>†</sup>, Carmelo Sferrazza<sup>†</sup>, Karen Liu<sup>†</sup>, Rocky Duan<sup>†</sup>, Guanya Shi<sup>†</sup>
</p>

<p align="center">
  <strong>
    <a href="https://rpl-humanoid.github.io/">Project Website</a> |
    <a href="https://arxiv.org/abs/2602.03002">Paper</a> |
    <a href="https://youtu.be/HLWY-KYt4gk">Video</a>
  </strong>
</p>

![](assets/website_hero.gif)

> [!NOTE]
> **Research release only.** For potential issues and future support, please open an issue here or directly contact Yuanhang Zhang (yuanhang0610@gmail.com).
> Any integration of RPL features into [Holosoma](https://github.com/amazon-far/holosoma) mainline is a separate effort and will not be reflected here. We use IsaacGym as the training backend since IsaacSim is not fully supported yet (force perturbation not implemented in IsaacSim API).

Code for RPL - per-terrain experts RL training, multi-depth distillation, and sim2sim, built on [Holosoma](https://github.com/amazon-far/holosoma).

Capabilities:

- **Bidirectional Multi-Terrain Locomotion** - walks forward and backward across stairs, stepping stones, and slopes.
- **Whole-Body Teleoperation** - tracks commanded base velocity and end-effector pose, with base height and torso orientation adjustable during stance.
- **Force-Adaptive Payload Carrying** - compensates for unknown external forces at the end effectors.

## Repo layout

| Path | What it holds |
| --- | --- |
| [`thirdparty/holosoma/`](https://github.com/amazon-far/holosoma) | Holosoma core (git submodule): warp ray-cast multi-depth rendering, PPO-MA, DAgger-MA, ViT / CNN modules, terrains |
| `rpl_training/` | Everything outside Holosoma core: loco-manipulation env, manager terms, presets, launchers, tests |
| `rpl_training/data/` | ACCAD upper-body motion clips and the four teachers (`models/`) |
| `rpl_sim2sim/` | Sim2sim: MuJoCo environment (sim side), the depth student for `holosoma_inference` (policy side), reference student exports |

## Setup

Linux with NVIDIA GPUs.

```bash
git clone --recurse-submodules https://github.com/amazon-far/RPL.git
cd RPL
```

The [sim2sim demo](#sim2sim-mujoco) and [training](#training) each install their own conda envs.

## Sim2sim (MuJoCo)

https://github.com/user-attachments/assets/e0c3e96a-239f-4308-b16d-7e270d708463

Runs the depth student in the MuJoCo sim2sim environment in
`rpl_sim2sim/` ([video](assets/rpl_sim2sim.mp4)). The sim and the policy run
as two processes communicating over Unitree `sdk2py` (DDS), with depth images passed through shared memory.

### 1. Install

```bash
bash rpl_sim2sim/setup_env.sh
```

This builds two conda envs under `~/.holosoma_deps/miniconda3` (override with `CONDA_ROOT`):

| Env | Role | Override |
| --- | --- | --- |
| `rplmujoco` | Sim: MuJoCo, depth renderer, Xbox pad, Unitree SDK bridge | `SIM_ENV` |
| `rplinference` | Policy: `holosoma_inference` + the RPL depth student | `POLICY_ENV` |

The script needs an existing conda at `CONDA_ROOT`. If you have none, run
`thirdparty/holosoma/scripts/setup_mujoco.sh` once first.

### 2. Launch

> [!TIP]
> We highly recommend an **Xbox** controller for sim2sim. Plug it into the host running `run_sim.sh`.
> The keyboard works, but it's harder to steer smoothly.

Start the sim first, then the policy, each in its own terminal:

```bash
# Terminal 1: sim (terrain course, Xbox pad on this host)
bash rpl_sim2sim/run_sim.sh

# Terminal 2: policy (shipped student)
bash rpl_sim2sim/run_policy.sh
```

| Goal | Command |
| --- | --- |
| Run your own student | `run_policy.sh --task.model-path /path/model_XXXXX.onnx` |
| Use the keyboard instead of an Xbox pad | `run_sim.sh --no-joystick` (see [Controls](#controls)) |

`run_sim.sh --help` and `run_policy.sh --help` list the rest.

### 3. Walk

1. Release the elastic band: press `9` in the MuJoCo window.
2. Start the policy right away: `start` on the pad (`]` on the keyboard).
3. Drive with the sticks (`w/s/a/d`, `q/e`). Stand/walk mode switches automatically
   from commanded velocity: walking when XY speed exceeds 0.1 m/s or absolute yaw
   rate exceeds 0.1 rad/s, stance otherwise. The existing model observation layout
   is preserved; no retraining is needed to use automatic switching.

### Controls

**Policy, Xbox pad:**

<table>
<tr>
<td>

| Input | Action |
| --- | --- |
| `start` | Start the policy |
| `A` + `X` | Init pose |
| Left stick | Forward speed |
| Right stick | Yaw |
| `L2` | Zero velocity, disable fixed speed, and enter stance |
| `B` + `Y` | Stop |
| `B` + d-pad up / down | Base height |
| `Y` / `select` + d-pad | Waist |
| `R1` + d-pad | Hands forward / back, apart / together |
| `X` + d-pad up / down | Hands up / down |

</td>
<td><img src="assets/xbox_controls.svg" alt="Xbox controller layout" width="400"></td>
</tr>
</table>

<details>
<summary><b>Policy, keyboard</b> (instead of the Xbox pad)</summary>

Run `run_sim.sh --no-joystick` and
`run_policy.sh --task.velocity-input keyboard --task.state-input keyboard`:

| Key | Action |
| --- | --- |
| `]` | Start the policy |
| `i` | Init pose |
| `w` / `s` / `a` / `d` | Linear velocity |
| `q` / `e` | Angular yaw velocity |
| `z` | Zero velocity and enter stance |
| `o` | Stop |

</details>

**MuJoCo window:**

| Key | Action |
| --- | --- |
| `9` | Release the elastic band |
| `7` / `8` | Band length |
| `3` - `6` | Move the band anchor |
| `R` | Toggle video recording (`logs/sim2sim/simulation.mp4`) |

## Training

### Install

```bash
bash rpl_training/training_runs/setup_env.sh
```

This builds the IsaacGym conda env `hsgym`. Optional environment variables:

| Variable | Effect |
| --- | --- |
| `ENV_NAME` | Conda env name (default `hsgym`) |
| `HOLOSOMA_LOCAL_DIR` | Use another Holosoma checkout instead of `thirdparty/holosoma` |

The launchers log to W&B by default. Set `LOGGER=disabled` to keep logs local under `logs/`, or
`WANDB_PROJECT` / `WANDB_ENTITY` / `RUN_NAME` to pick the W&B destination. `NGPUS` sets the number of GPUs,
and extra arguments pass through to `rpl_training.train_agent`.

For multi-GPU IsaacGym runs, the launchers build/load `gymtorch` once before
starting the workers. The first launch may pause while compiling; later launches
reuse the cached extension. IsaacGym launchers default to
`TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions/rpl_hsgym_front"` and
`MAX_JOBS=2`. Both the preparation step and workers inherit these settings;
explicit environment values override the defaults.

### 1. Train the RL experts
We recommend training the experts on 4 GPUs.
```bash
NGPUS=4 EXPERT=stones bash rpl_training/training_runs/run_expert.sh --training.name stones-expert
NGPUS=4 EXPERT=stairs bash rpl_training/training_runs/run_expert.sh --training.name stairs-expert
NGPUS=4 EXPERT=slopes bash rpl_training/training_runs/run_expert.sh --training.name slopes-expert
```

### 2. Distill the multi-depth student
We recommend training the student on 8 GPUs.
```bash
NGPUS=8 bash rpl_training/training_runs/run_distill.sh --training.name rpl-student
```

It loads the shipped teachers
by default. To distill your
own experts, pass either:

```bash
# A directory with the same per-terrain layout as rpl_training/data/models
--algo.config.ckpt-teacher DIR

# Or one W&B checkpoint per terrain (slopes, stairs_up, stairs_down, stepping_stones_columns)
--algo.config.ckpt-teacher-wandb.slopes wandb://ENTITY/PROJECT/RUN_ID/model_XXXXX.pt
```

Distillation runs export `model_XXXXX.onnx` next to each checkpoint for [sim2sim](#sim2sim-mujoco).

### Front-camera-only distillation on four GPUs

The front-only preset uses the shipped terrain teachers and starts a new student with
one torso-mounted front depth camera. It retains the 48×27 depth images, two-frame
history, 10 Hz camera updates, noise and pose randomization. The existing dual-camera
presets remain available.

On the four-GPU Ada machine, after the training environment setup above:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPUS=4 \
  bash rpl_training/training_runs/run_distill_front.sh \
  --training.name rpl-front-4gpu
```

The launcher defaults to four GPUs and the `g1-29dof-rpl-distill-front` experiment:
4,096 environments total (1,024 per GPU), four minibatches, and 20,000 learning
iterations. This keeps the dual-camera preset's per-GPU environment count; the total
batch is half its eight-GPU run. GPU memory use and throughput still need measuring
on your machine. To start with fewer environments, append `--training.num-envs 2048`
(512 per GPU). Set `LOGGER=disabled` for local logging. Do not pass a dual-camera
student checkpoint to resume this run; the teacher checkpoints are reused automatically.

To evaluate the resulting single-camera ONNX export, run these in separate terminals:

```bash
bash rpl_sim2sim/run_sim.sh --config rpl_sim2sim/config/loco_manip_cam_front.yaml
bash rpl_sim2sim/run_policy.sh inference:g1-29dof-rpl-depth-front \
  --task.model-path /absolute/path/to/model_XXXXX.onnx
```

Supply your front-only export explicitly; the shipped ONNX models use two cameras.

## Agibot X2 Ultra

X2 includes a head-fixed 29-joint robot model, G1-derived expert/distillation
presets, a four-GPU training launcher and MuJoCo sim2sim. The commands above
continue to select G1. For X2, use:

```bash
TASK=stairs bash rpl_training/training_runs/run_x2.sh
```

X2 defaults to four GPUs for stones/stairs and one GPU for flat ground/slopes. The dual-camera
student uses four GPUs with 2,048 environments each (8,192 total), preserving the
reference student's global rollout size.

See the [X2 runtime guide](docs/x2_ultra_training_review.md) for teacher setup,
sim2sim and single-GPU smoke tests, plus the [parameter reference](docs/x2_ultra_parameters.md).

## Citation

```bibtex
@article{zhang2026rpl,
  title   = {RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains},
  author  = {Zhang, Yuanhang and Seo, Younggyo and Chen, Juyue and Yuan, Yifu and Sreenath, Koushil and Abbeel, Pieter and Sferrazza, Carmelo and Liu, Karen and Duan, Rocky and Shi, Guanya},
  journal = {arXiv preprint arXiv:2602.03002},
  year    = {2026}
}
```

Related: [FALCON](https://lecar-lab.github.io/falcon-humanoid/)

## License

Apache-2.0, see [`LICENSE`](LICENSE).
