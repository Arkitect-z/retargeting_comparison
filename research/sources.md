# First-party sources

Accessed: 2026-07-22

Every Stage 0 classification below is anchored to an official paper, project page, repository, or documentation site.
Issues and third-party summaries are not primary evidence.

- [Controlled Sparse-IK](https://github.com/kevinzakka/mink)
- [Controlled Dense-KeyBody IK](https://github.com/kevinzakka/mink)
- [GMR](https://github.com/YanjieZe/GMR)
- [OmniRetarget / Holosoma](https://github.com/amazon-far/holosoma)
- [ProtoMotions v3](https://github.com/NVlabs/ProtoMotions)
- [PHC retargeter](https://github.com/ZhengyiLuo/PHC)
- [ProtoMotions v2](https://github.com/NVlabs/ProtoMotions)
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
- [cuRoboV2 MotionRetargeter](https://nvlabs.github.io/curobo/latest/getting-started/humanoid_retargeting.html)
- [MaskedMimic](https://research.nvidia.com/labs/par/project/maskedmimic.html)
- [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco)
- [Mink](https://github.com/kevinzakka/mink)
- [PyRoki](https://github.com/chungmin99/pyroki)
- [MIRROR](https://github.com/ami-iit/paper_ramadoss-2022-ral-humanoid-retargeting)
- [ReActor](https://arxiv.org/abs/2605.06593)
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
