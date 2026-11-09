# G1 ACCAD to X2 retargeting

The [X2 library](../rpl_training/data/motions/x2/falcon_accad_all/) contains
252 ACCAD clips (48,233 frames) retargeted from the packaged G1 library to RPL's
29-DOF, fixed-head X2 model. Every clip preserves its source frame count and
30 fps rate, so its duration is `(frames - 1) / 30` seconds.

## Pipeline

[scripts/retarget_g1_accad_to_x2.py](../scripts/retarget_g1_accad_to_x2.py) uses the
local `robot_retargeter` kit for forward kinematics, link scaling, contact
extraction/locking, and Mink IK with DAQP. Source data and kit files are read-only.
The recorded kit revision is in the
[provenance manifest](../rpl_training/data/motions/x2/retarget_manifest.json).

The script adds auxiliary markers to copies of RPL's G1 and X2 models, preserving
training kinematics and X2 joint ordering. Foot markers use the sole planes
(-35 mm for G1, -73 mm for X2). The fixed-head X2 model omits the head IK task.
Elbow/wrist position task costs default to 100; kit knee/elbow safety margins
remain enabled.

After IK, a symmetric forward/backward projection bounds joint speeds to 95% of
authored limits at the source frame rate. The exporter recomputes X2 body poses
and all velocities from the resulting trajectory. NPZ files use the Holosoma
schema with explicit joint/body names, root quaternions in xyzw order, and body
quaternions in wxyz order. The full-body trajectory supplies kinematic context;
the upper-body training command consumes the 14 arm channels.

Ten-second duration normalization is an optional motion-loader setting used by
G1 training. It does not belong in dataset export. X2 training presets use this library with G1's ten-second duration normalization.
Only the 14 arm channels supply training references; root and leg trajectories
are not replayed. The optional flat stage has no hand forces; experts and the
depth student enable them.

## Generate and validate

Run from the repository root using the kit's existing Python environment, which
contains MuJoCo and Mink. Retargeting runs on CPU and takes several minutes with
four workers on this machine.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/ubuntu/anaconda3/envs/robot_retargeter/bin/python \
  scripts/retarget_g1_accad_to_x2.py \
  --kit /home/ubuntu/robot_retargeter \
  --workers 4 --resume \
  --manifest rpl_training/data/motions/x2/retarget_manifest.json

OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/anaconda3/envs/robot_retargeter/bin/python \
  scripts/validate_x2_motion_library.py

OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/anaconda3/envs/robot_retargeter/bin/python \
  scripts/test_retarget_g1_accad_to_x2.py
```

The default work directory is `logs/retargeting/g1_to_x2_accad/final/`. It holds
adapted models/configs, keypoints, per-clip logs/metrics, raw IK CSVs under
`raw_robot_motion/`, and a run manifest. CSV root quaternions are xyzw; replay
requires the generated `x2_fk.xml` or RPL's 29-DOF asset. The kit's default
31-DOF model has a different joint layout.

`--resume` reuses clips only when generator, robot models/configs, kit revision,
source hash, and output hash match. Missing or invalid cached metrics trigger
regeneration. `--clips` and `--limit` select smaller runs; use a separate output
and work directory for these, and reserve the library `--manifest` path above
for full runs.

Validation compares every output with its source frame rate/count and checks
hashes, array shapes, finite values, quaternion norms, joint ordering, joint
limits, speed limits, and joint position/velocity agreement. Regression tests
also cover FK, world angular velocities, fractional frame rates, and resume
invalidation. These checks establish kinematic consistency, not policy tracking
or dynamic balance.

### Data verification (2026-11-09)

The corrected library passed checks for every clip and frame:

- All 252 outputs retain 30 fps and all 48,233 source frames; source/output hashes match the manifest.
- All stored arrays are finite, and joint/body ordering and quaternion norms match the X2 schema.
- Maximum joint-limit roundoff is 1.05e-8 rad; maximum joint speed is 0.950001 of its authored limit.
- FK recomputed with the canonical X2 model agrees with saved body positions within 4.47e-7 m and rotations within 2.68e-7 rad.
- Independently recomputed joint/body velocities agree within 1.16e-5 in their respective units.
- The production motion loader passed 48,989 samples covering every interpolation interval, endpoints, and out-of-range clamping. It returns 29 joint channels, including the 14 arm-reference channels. Maximum float32 interpolation differences from a float64 calculation are 3.35e-5 rad and 0.00102 rad/s.

The files are suitable as kinematic upper-body motion references with native
playback timing. This verification does not establish full-body dynamic balance
or successful trained-policy tracking. Speed projection changes some poses:
131 clips contain adjustments, and the largest joint correction is 0.997 rad.
The manifest records per-clip fitting errors and projection statistics.

Reproduce the additional checks without changing training code or motion files:

```bash
# Canonical-model FK and independent velocity checks for every frame.
OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/anaconda3/envs/robot_retargeter/bin/python \
  scripts/validate_x2_motion_library.py --kinematics \
  --report logs/retargeting/g1_to_x2_accad/kinematics-validation.json

# Loader compatibility only; this does not start a simulator or training job.
source scripts/source_isaacgym_setup.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/validate_x2_motion_library.py --loader \
  --report logs/retargeting/g1_to_x2_accad/loader-validation.json
```

The loader check uses `standardize_motion_length=False` to retain the native
timing that the saved velocities describe. Training-time normalization
to ten seconds changes playback speed and is outside this native-timing data
verification. The shared G1 loader does not rescale stored velocities during
normalization; X2 retains that behavior for training parity.

## Optional visual review

[Watch the regenerated arm-reference comparison](../assets/x2_accad_retarget_demo.mp4).
It uses the corrected 30 fps motion library and presents ten clips over
100 seconds. Each clip is resampled to ten seconds for this video, so video
playback speed differs from native motion timing when a source clip is shorter
or longer than ten seconds.

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/ubuntu/anaconda3/envs/robot_retargeter/bin/python \
  scripts/render_retargeted_accad.py --clips 0 120 217
```

The renderer writes full-body and arm-reference comparisons under the work
folder's `videos/` directory. G1 appears on the left and X2 on the right; root XY
translation is centered. For the arm comparison, both robots use stationary
HOME legs and torso. Playback defaults to ten seconds per clip at 30 video fps;
`--seconds` changes presentation speed without changing NPZ timing. Use
`--thumbnail-only` for a quick geometry check. Rendering requires EGL/OpenGL
and ffmpeg. Intermediate videos stay under ignored `logs/`; the arm-reference
comparison is also available under `assets/` for viewing from the README.
