# First-party sources

Accessed: 2026-07-22

Every Stage 0 classification below is anchored to an official paper, project page, repository, or documentation site.
Issues and third-party summaries are not primary evidence.

- [Controlled Sparse-IK](https://github.com/kevinzakka/mink) — frozen evidence: `benchmark v2`
- [Controlled Dense-KeyBody IK](https://github.com/kevinzakka/mink) — frozen evidence: `benchmark v2`
- [GMR](https://github.com/YanjieZe/GMR) — frozen evidence: `bb1bbe40774794fceb2a7c579a3464a28e68c844`
- [OmniRetarget / Holosoma](https://github.com/amazon-far/holosoma) — frozen evidence: `5f48635a3624656a5f46a07df26d43187e59f855`
- [ProtoMotions v3](https://github.com/NVlabs/ProtoMotions) — frozen evidence: `49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c`
- [PHC retargeter](https://github.com/ZhengyiLuo/PHC) — frozen evidence: `846988d433ce1f341e85ac6fbd2cd51911bb3341`
- [ProtoMotions v2](https://github.com/NVlabs/ProtoMotions) — frozen evidence: `historical ref not frozen`
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) — frozen evidence: `accessed 2026-07-22`
- [cuRoboV2 MotionRetargeter](https://nvlabs.github.io/curobo/latest/getting-started/humanoid_retargeting.html) — frozen evidence: `accessed 2026-07-22`
- [PhySINK / PHUMA](https://davian-robotics.github.io/PHUMA/) — frozen evidence: `paper/project evidence`
- [MaskedMimic](https://research.nvidia.com/labs/par/project/maskedmimic.html) — frozen evidence: `official project`
- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) — frozen evidence: `official repository`
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) — frozen evidence: `official continuous project`
- [Mink](https://github.com/kevinzakka/mink) — frozen evidence: `1.1.1 in benchmark env`
- [PyRoki](https://github.com/chungmin99/pyroki) — frozen evidence: `ProtoMotions frozen dependency`
- [MIRROR](https://github.com/ami-iit/paper_ramadoss-2022-ral-humanoid-retargeting) — frozen evidence: `official THEMIS implementation`
- [ReActor](https://arxiv.org/abs/2605.06593) — frozen evidence: `paper evidence`
- [GMR paper](https://arxiv.org/abs/2510.02252)
- [OmniRetarget paper](https://arxiv.org/abs/2509.26633)
- [H2O / PHC project](https://human2humanoid.com/)
- [PHUMA project](https://davian-robotics.github.io/PHUMA/)
- [Gleicher 1998](https://graphics.cs.wisc.edu/Papers/1998/Gle98/)
- [Gleicher 1997](https://graphics.cs.wisc.edu/Papers/1997/Gle97a/)

Exact repository revisions and local license observations are frozen in `manifests/repositories.csv`.

## LAFAN1 availability note

The official Ubisoft repository was accessed first, but its Git LFS endpoint reported an exhausted budget for the official archive (SHA-256 `ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`).
The Pilot uses the per-file mirror frozen in `manifests/dataset.yaml`; it matches the official 77 filenames, 496,672 frames, and nominal 30 fps and is never represented as an official Ubisoft host.
