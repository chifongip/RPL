# X2 runtime validation

Validated on 2026-11-06 on the current computer's NVIDIA GeForce RTX 4090 Laptop GPU (16 GiB). Training uses `hsgym` (Python 3.8, PyTorch 2.4.1+cu121); MuJoCo and inference use `rplmujoco` and `rplinference` (Python 3.10). MuJoCo version is 3.10.0.

These checks establish runtime/model consistency and finite short-run optimization/inference. Three-iteration smoke checkpoints are not usable walking policies. Full training has not been launched.

## Motion and hand-force integration (2026-11-09)

The complete bounded suite passed again under
`logs/x2-runtime-smoke/20261109_065947/` with X2 ACCAD references enabled in
all training stages, hand forces enabled for experts/student, G1 flat command
ranges restored, and nominal commanded height set to 0.64 m. The height-scan
offset and geometric initialization remain 0.68 m. Motion playback uses G1's
10-second normalization and unchanged stored-velocity behavior.

- Training regression suite: 120 passed, 2 skipped; simulator/model/derivation
  suite: 26 passed; policy suite: 28 passed.
- Real-clip command checks cover arm ordering, normalized playback, frozen
  references, upper-body resets and motion-end termination. A deterministic
  nonzero-scale force check covers wrist bodies, arm budgets, application
  positions and observation buffers independently of initial curriculum gating.
- Flat PPO, terrain-expert PPO and depth DAgger each completed three iterations
  with 32 environments. New smoke expert weights supplied all four student
  teacher routes. Maximum PyTorch/ONNX errors were 3.35e-8, 2.61e-8 and 5.22e-8,
  respectively.
- Both flat and dual-depth simulator/policy runs passed 50 policy ticks over
  DDS, including init/start/stop; the depth run consumed both camera streams.
  Inference still supplies HOME arm references with interactive IK disabled.

The earlier results below record the original fixed-reference integration.
These new smoke results establish integration, not tracking quality or balance.

## Hand sampling segment update (2026-11-09)

X2 now samples between wrist-roll-local `(0, 0, -0.03)` and `(0, 0, -0.15)` m,
inside the simplified hand collision box. X2 uses the existing extension offset
and ratio range `(0.2, 1.0)`; the shared force manager is unchanged from G1.
All 41 targeted X2 runtime/motion/force and G1 command-parity tests pass,
including segment endpoints, interior samples and rotation into world
coordinates. The updated comparison is in the [training guide](x2_ultra_training_review.md).

The last full GPU smoke used the earlier equivalent geometry implementation.
The run under `logs/x2-runtime-smoke/20261109_080956/` passed 120 training
tests (2 skipped), 26 simulator/model tests and 28 policy tests. All three
three-iteration training stages and export checks passed; maximum ONNX error
was 4.48e-8. The first depth transport process wrote its successful 50-tick
report, then exited with SIGSEGV during shutdown. A separate flat/depth transport
rerun with the same exports passed with clean process exits; reports are under
`transport-rerun/`. The original shutdown failure was not reproduced or resolved
by this geometry change.

## Learning algorithm boundary

X2 uses the corresponding G1 PPO trainer, symmetry utility, action term and environment classes directly. There is no X2 PPO implementation, custom sample mask, loss normalization or distributed-loss validity guard. The Holosoma submodule has no working-tree changes. X2's DAgger adapter only selects and validates teacher checkpoints; it does not change the distillation update.

Generated simulation assets use symmetric waist-yaw hard stops at `[-2.2078, 2.2078]` rad. The standard 95% soft-limit rule gives `[-2.09741, 2.09741]` rad. The packaged URDF, converted MuJoCo model and runtime robot configuration agree. Only this joint's position limits are adapted. Original vendor files and their `[-3.43, 2.2078]` range remain unchanged, and the parameter reference retains their original values. All source/reference hashes match the committed preparation snapshot.

## Results

| Check | Result |
| --- | --- |
| Training regression suite | 100 passed, 2 skipped; MuJoCo-dependent checks run separately below |
| MuJoCo/model/derivation suite | 25 passed |
| Policy suite | 27 passed |
| Algorithm/configuration parity | Complete PPO configuration, action configuration and environment class match corresponding G1 presets; learning settings and global rollouts retained |
| Joint ordering | Native X2 waist/wrist order checked through named body groups, per-joint PD inputs, observations, bridge states/commands and exports |
| Symmetry | Standard G1 transforms preserve HOME, reflect every joint according to its physical angular axis, and are involutions; adapted position limits match their reflected counterparts |
| IsaacGym import | 30 bodies, 29 joints; native names/order and every position, velocity and effort limit accepted |
| IsaacGym mass | 41.96652195 kg total; 11.86347294 kg torso including fixed-head mass |
| Joint dynamics | Authored friction 0.3 and damping 0 verified; per-DOF armature 0.03 specified |
| Flat PPO smoke | 32 environments, 3 iterations using G1 PPO; finite losses, checkpoint and ONNX produced |
| Terrain-expert PPO smoke | 32 environments, 3 iterations using G1 PPO; all terrain types retained, checkpoint and ONNX produced |
| Depth DAgger smoke | 32 environments, 3 iterations; all four X2 teacher routes loaded, checkpoint and ONNX produced |
| Export checks | Joint names/partitions, simulation position limits, HOME and ordered gains verified; deterministic PyTorch/ONNX outputs agree |
| Live flat sim2sim | 1,600 physics steps and 50 policy ticks through DDS; init/start/stop passed |
| Live depth sim2sim | Same bounded run; both shared-memory camera streams consumed; init/start/stop and clean shutdown passed |
| Depth processing | NumPy antialiased bicubic resize agrees with training's Torch implementation |
| Compatibility | G1's legacy actuator layout still maps; X2 rejects G1 teachers and inference exports |
| End-effector rewards | X2 uses final wrist-roll links; G1 wrist-yaw fallback retained |
| Failure/cleanup checks | Camera errors or missing frames after a successful frame reach the main process; duplicate producers are rejected without altering the owner; all cameras and shared memory close even after renderer cleanup errors; support forces clear when the band is released |
| Viewer startup | EGL priming detects depth cameras with video disabled; headless bypass tested with a mocked renderer |

Maximum absolute PyTorch/ONNX errors:

- Flat: `5.58793545e-08`.
- Terrain expert: `3.25962901e-08`.
- Depth student: `6.33299351e-08`.

Flat actor inputs were `[1,575]` for both actors. Depth lower/upper inputs were `[1,5761]` / `[1,575]`; depth shape was `[2,1,27,48]`, produced at approximately 10 Hz. Both policy runs used HOME arm references with G1 IK disabled.

The complete smoke run is recorded locally under `logs/x2-runtime-smoke/20261106_014554/`. The final review confirms that all adapted joint-position limits reflect to their partners, and PPO settings are inherited directly without a redundant X2 override. Logs are ignored by Git; commands are in the [runtime guide](x2_ultra_training_review.md). Ruff, shell syntax and Git whitespace checks pass.

## Training allocation and remaining validation

- Flat ground and slopes default to one GPU and 4,096 total environments each. Stones/stairs default to four GPUs and 16,384 total environments.
- The student retains 8,192 total environments and 24 steps per environment: 196,608 transitions per global rollout on four GPUs, or 2,048 environments per GPU. The reference eight-GPU run has the same global rollout. Learning settings and iteration counts are unchanged.
- Full-size memory use, throughput and four-GPU execution remain untested on the target machine. All local training smokes explicitly use 32 environments and one GPU.
- The smoke student uses one short X2 expert for every teacher route. Full distillation requires trained X2 terrain experts.
- Startup and transport tests use a support band. HOME joint PD alone does not balance the untrained floating robot. Walking, command tracking and terrain traversal require trained-policy evaluation; the bundled MuJoCo scene is flat.
- Camera transforms, resizing and delay are checked; full image equivalence and trajectory/contact equivalence between physics engines are not established. Training retains camera randomization while MuJoCo uses the nominal rig.
- The narrowed negative waist-yaw stop is a simulation adaptation. Later evaluation against the original asymmetric model is needed before hardware deployment, which remains outside the current stage.
