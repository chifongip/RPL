# Agibot X2 Ultra: preparation status and implementation plan

**Current stage: parameter derivation is complete for the proposed simulation baseline. Runtime integration has not started.** The repository still launches G1 experiments and G1 sim2sim. No persistent X2 29-joint assets, training/inference presets, trained policies or X2 launch commands have been added.

The scope is IsaacGym training and MuJoCo sim2sim. Hardware communication, identification, physical camera calibration and real-robot trials are deferred.

## Documentation

| Document | Purpose | How to update |
| --- | --- | --- |
| This page | Agreed scope, current evidence and implementation sequence | Edit directly as work progresses |
| [Parameter reference](x2_ultra_parameters.md) | Ordered joint table, geometry, camera transforms and proposed simulation settings | Regenerate with the derivation script |
| [Parameter JSON](x2_ultra_parameters.json) | Full-precision arrays, merged inertias, mesh transforms and source hashes | Regenerate with the derivation script |

The generated files record model-derived values separately from inherited or proposed settings. Timing, action scaling, friction settings, reward thresholds and randomization ranges are simulation assumptions or task choices; robot geometry does not determine them.

## Agreed robot definition

| Item | Decision |
| --- | --- |
| Robot | Agibot X2 Ultra from `/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.3.0` |
| Head | Fix yaw and pitch at zero, preserving head mass, inertia and geometry |
| Actions | 29: 12 leg + 3 waist joints in the lower body; 14 arm joints in the upper body |
| PD gains | `X2_GAIN_PRESETS["agibot_stiff"]` from the reference constants file below |
| Standing pose | `HOME_KEYFRAME` from the same file, including nominal pelvis height 0.68 m |
| Cameras | G1 dual front/back torso-mounted training rig, with X2 meshes for self-occlusion |
| Torso commands | Preserve RPL's whole-body interpretation and initial range/curriculum; assess feasibility through simulation |

The gain/pose reference is `/home/ubuntu/unitree_rl_mjlab/src/assets/robots/agibot_x2/x2_constants.py`. Only the named PD gains and standing pose are adopted from it. All numerical values are listed in the parameter reference rather than repeated here.

Despite the name `waist_pitch`, RPL rewards desired torso tilt through projected gravity. Hip/leg and pelvis motion can compensate for limited waist travel. Do not clamp torso pitch commands to X2's waist-pitch joint limits. The yaw reward separately tracks the waist-yaw joint.

### Camera mounting preview

![X2 with the selected front and back torso camera mounts and fields of view](assets/x2_camera_mounts.png)

The MuJoCo capture uses HOME and the derived camera transforms. Blue marks the front camera and orange the back camera. Housings and frustum lengths are illustrative; mount positions and viewing axes follow the selected simulation rig. Both cameras sit near the lower torso and look 75 degrees downward from horizontal.

Regenerate with `/home/ubuntu/.holosoma_deps/miniconda3/envs/rplmujoco/bin/python scripts/render_x2_camera_mounts.py` on a machine with EGL rendering support. This preview creates the head-fixed model in memory and does not modify runtime assets or presets.

## Source assets and derived findings

The simplified-collision URDF is the proposed training geometry source; MuJoCo geometry must be checked against it during integration. Vendor files remain unchanged.

| Source asset | Relevant findings |
| --- | --- |
| `x2_ultra.urdf` | Full model: 41 links and 31 moving joints |
| `x2_ultra_simple_collision.urdf` | 39 links, the same moving-joint transforms/limits; collision geometry mixes meshes and primitives |
| `x2_ultra.xml` | Vendor MuJoCo model used for independent forward-kinematics checks and the in-memory head-exclusion test |

The simplified URDF omits the full model's fixed `base_link` and `lidar_chest_front` frames. The local `x2_ultra_simplified.xml` is an alternative asset and is not the source for the parameter derivation. The `x2_fist` files describe a different robot variant.

Key derived findings:

- Head-fixed model: 29 articulated joints, an expected 30 collapsed robot bodies, and MuJoCo dimensions `nq=36`, `nv=35`, `nu=29`. IsaacGym's actual loaded body ordering remains to be checked.
- URDF mass is 41.966521 kg, including 1.719334 kg of preserved head mass. The vendor XML differs by approximately 0.000048 kg; tests allow that source difference.
- Each simplified foot has 12 collision spheres, with 10 at the lowest sole level. At HOME, the lowest surfaces start approximately 13.49 mm above flat ground. Keep the selected initialization and validate settling; the calculated contact height is not a tested standing equilibrium.
- X2 waist and wrist joint order differs from G1. After excluding head joints and motors, the vendor XML's remaining joint and actuator orders agree with the URDF traversal. Runtime mappings still need verification.
- The existing G1 sim2sim camera offsets and delay differ slightly from the selected G1 training rig. X2 sim2sim must explicitly reproduce the selected rig and its optical-frame conversion.

## Validation completed

The [derivation script](../scripts/derive_x2_parameters.py) and [seven tests](../rpl_training/tests/test_x2_parameters.py) check:

- Regeneration of both numerical artifacts from the local sources.
- Gain coverage, HOME within hard joint limits, and head-excluded joint/actuator ordering.
- Physical source inertia tensors and mass conservation through fixed-link merging.
- Foot geometry, HOME mirror consistency and conversion between action offsets and joint limits.
- Camera rotation orthonormality and the analytical Warp-to-MuJoCo viewing-axis conversion.
- Independent HOME forward kinematics and COM against the original vendor MuJoCo model.
- Successful compilation of an in-memory head-fixed MuJoCo model, with 29 actuators and preserved mass.

These are derivation and model-structure checks. Settled standing, IsaacGym import, rendered camera parity, training, policy export and closed-loop sim2sim have not been tested. The in-memory model is not a saved training asset.

To reproduce from the repository root using the existing local environment:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/rplmujoco/bin/python scripts/derive_x2_parameters.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ubuntu/.holosoma_deps/miniconda3/envs/rplmujoco/bin/python -m pytest rpl_training/tests/test_x2_parameters.py -q -p no:cacheprovider
```

Alternatively, activate an environment with NumPy, pytest and MuJoCo and use `python`. The script accepts `--assets`, `--constants` and `--output` for different local paths. Tests use the default local source paths and skip derivation checks when those sources are absent; MuJoCo-dependent checks skip when MuJoCo is unavailable. The JSON stores local source paths and hashes, so regeneration on another checkout can change provenance.

## Remaining implementation and acceptance checks

| Step | Work | Required evidence |
| --- | --- | --- |
| 1. Assets and robot configuration | Save head-fixed URDF/XML derivatives and required meshes/license; apply derived names, limits, gains, HOME and simulation baseline | IsaacGym loaded names/limits match configuration; head mass is retained; collision geometry is consistent across simulators |
| 2. Flat-ground baseline | Add an X2 experiment with fixed HOME arm references (`motion=None`), zero torso/height offsets initially, no hand forces and no depth input | Stable reset/settling, finite observations/actions, functioning PD/torque clipping and a short training smoke test |
| 3. Export and early sim2sim | Add X2 inference/layout/joint mappings and simulate an exported flat-ground policy in MuJoCo; keep interactive G1 arm IK disabled | PyTorch/export agreement on identical inputs; controlled standing and forward/backward/yaw behavior; repeatable reset/stop behavior |
| 4. Terrain experts | Add X2 terrain presets and introduce torso/base-height curricula and randomization progressively | Command tracking, fall rate, joint limits, torque saturation and reachability evaluated on X2 |
| 5. Depth distillation | Use the selected dual-camera rig with X2 self-occlusion and newly trained X2 teachers | Camera geometry/depth processing agree across simulators; exported student evaluated on held-out terrain |

Use the proposed numerical baseline to begin implementation; no additional robot parameters are needed from the user at this stage. Validate the assumptions as each step runs. Reuse G1's algorithms, observation organization, terrain generation and curriculum mechanisms, with X2-specific geometry and mappings.

G1 checkpoints and joint-space motion clips are incompatible with direct X2 use. Retargeting motion clips, interactive hand IK and hand-force tasks are optional later extensions, not requirements for the fixed-arm baseline or its terrain teachers. Depth distillation needs X2 checkpoints for all four routing keys: `slopes`, `stairs_up`, `stairs_down` and `stepping_stones_columns`; a shared stairs policy must be supplied explicitly for both stairs keys.

Primary integration files are `rpl_training/config_values/{robot,experiment,registry,camera}.py`, the training launchers, and `rpl_sim2sim/{inference_presets,policy,bridge,simulator}.py` with new assets/configs. Existing G1 launch commands remain G1-only until that work is implemented.
