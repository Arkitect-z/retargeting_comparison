# ProtoMotions v3 trajectory-length protocol correction

Date frozen: 2026-07-23

## Finding

ProtoMotions v3's official PyRoki workflow is a whole-trajectory optimizer, not
a frame-independent IK loop. The official documentation states that every
motion is trimmed or padded to 15 seconds, or 450 frames at 30 FPS, for JAX
compilation and batch processing:

<https://nvlabs.github.io/ProtoMotions/tutorials/workflows/retargeting_pyroki.html>

The frozen upstream script independently confirms
`--target-raw-frames` defaults to `450`. The benchmark had overridden this
argument to 600 to force full coverage of the 20-second Pilot. That override
changed the upstream fixed-shape operating point and was therefore a protocol
error. No 600-frame result may be labelled the official ProtoMotions v3
operating point.

## CUDA diagnosis

The 600-frame graph contained 5,996 cost terms and 1,800 variables. On the
RTX 3090 Ti, XLA rejected a generated kernel that requested 131,072 bytes of
shared memory when 101,376 bytes were available. This was a per-kernel
shared-memory resource failure, not a global VRAM OOM.

Restoring 450 frames reduced the graph to 4,496 cost terms and 1,350 variables
and removed that failure. The first corrected attempt then exposed a separate
benchmark-wrapper defect: `JAX_PLATFORMS=cuda` did not provide the CPU backend
required by PyRoki's `jax.debug.callback` logging. The corrected environment is
`JAX_PLATFORMS=cuda,cpu`; it keeps the solve on CUDA and permits CPU callbacks.

The corrected full-shape diagnostic succeeded with 450 finite G1-29 frames,
terminated normally after 103 solver iterations, and used approximately
6.1 GiB of 24.6 GiB VRAM during observation. Its one-call cold wall time was
505.26 seconds inside the formal in-memory boundary. This includes first-shape
JAX compilation and is not the steady-state warm timing result.

## Correct comparison contract

- Frozen source remains the preselected 600-frame LAFAN1 Pilot.
- Official ProtoMotions v3 consumes source frames `[0, 450)`.
- ProtoMotions v3 native-contract completion is `450/450 = 100%`.
- Its full-source coverage is `450/600 = 75%`; canonical full-source completion
  remains `incomplete`.
- Cross-method quality comparisons that include ProtoMotions v3 use the same
  source frames `[0, 450)` for every method.
- Other methods may additionally report their full 600-frame Pilot metrics.
- No padding, interpolation, or silent stitching may disguise the missing
  150 source frames.
- A future chunked 600-frame extension, if studied, must be labelled a
  benchmark extension rather than official ProtoMotions v3.

## Evidence disposition

The previous 600-frame CUDA and CPU artifacts are retained byte-for-byte as
protocol-deviation evidence and excluded from formal charts, ranks, timing
frontiers, scale-response publication, and Stage 2 cost projection. The
corrected formal output revision is `protomotions-v3-v3`.
