"""Audited LAFAN1-BVH to skinned-SMPL visualization adapter.

LAFAN1 is distributed as a 22-joint BVH skeleton and does not contain SMPL or
SMPL-X pose/shape parameters.  This module therefore treats the skinned body as
an explicit fitted visualization derivative.  It never represents the result
as dataset-native SMPL evidence.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, load_yaml, sha256_file
from .schemas import CanonicalHuman


SMPL_JOINT_MAP: tuple[tuple[str, int], ...] = (
    ("Hips", 0),
    ("LeftUpLeg", 1),
    ("RightUpLeg", 2),
    ("Spine", 3),
    ("LeftLeg", 4),
    ("RightLeg", 5),
    ("Spine1", 6),
    ("LeftFoot", 7),
    ("RightFoot", 8),
    ("Spine2", 9),
    ("LeftToe", 10),
    ("RightToe", 11),
    ("Neck", 12),
    ("LeftShoulder", 13),
    ("RightShoulder", 14),
    ("Head", 15),
    ("LeftArm", 16),
    ("RightArm", 17),
    ("LeftForeArm", 18),
    ("RightForeArm", 19),
    ("LeftHand", 20),
    ("RightHand", 21),
)

# SMPL's template uses Y-up.  The benchmark uses the same right-handed Z-up
# conversion as canonical LAFAN: [x, y, z] -> [x, -z, y].
SMPL_Y_UP_TO_CANONICAL_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def default_smpl_model_root(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root).resolve()
    return (root.parent / "body_models" / "smpl_chumpyfree").resolve()


def default_smpl_model_file(repo_root: str | Path = ".") -> Path:
    return default_smpl_model_root(repo_root) / "smpl" / "SMPL_NEUTRAL.pkl"


def default_skin_cache(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    return (
        root
        / "source_adapters"
        / "smpl_skin"
        / str(sequence["sequence_id"])
        / "lafan_bvh_fitted_smpl.npz"
    )


@dataclass(frozen=True)
class SmplSkinMotion:
    pose_aa_zup: np.ndarray
    betas: np.ndarray
    body_scale: float
    root_positions_zup: np.ndarray
    transl_zup: np.ndarray
    source_frame_idx: np.ndarray
    fps: float
    metadata: dict[str, Any]

    def validate(self) -> None:
        frames = len(self.pose_aa_zup)
        if self.pose_aa_zup.shape != (frames, 72):
            raise ValueError("Fitted SMPL pose must have shape [T,72]")
        if self.betas.shape != (10,):
            raise ValueError("Fitted SMPL betas must have shape [10]")
        if self.root_positions_zup.shape != (frames, 3):
            raise ValueError("Fitted SMPL roots must have shape [T,3]")
        if self.transl_zup.shape != (frames, 3):
            raise ValueError("Fitted SMPL translations must have shape [T,3]")
        if not np.array_equal(self.source_frame_idx, np.arange(frames)):
            raise ValueError("Fitted SMPL source-frame indices must be contiguous")
        if not (
            np.isfinite(self.pose_aa_zup).all()
            and np.isfinite(self.betas).all()
            and np.isfinite(self.root_positions_zup).all()
            and np.isfinite(self.transl_zup).all()
            and math.isfinite(self.body_scale)
            and self.body_scale > 0.0
            and math.isfinite(self.fps)
            and self.fps > 0.0
        ):
            raise ValueError("Fitted SMPL artifact contains invalid numeric values")
        if self.metadata.get("source_representation") != "LAFAN1_BVH":
            raise ValueError("Fitted SMPL artifact misstates the LAFAN source format")
        if self.metadata.get("dataset_native_smpl") is not False:
            raise ValueError("A LAFAN fitted mesh must not claim dataset-native SMPL")

    @classmethod
    def load(cls, path: str | Path) -> "SmplSkinMotion":
        with np.load(path, allow_pickle=False) as archive:
            value = cls(
                pose_aa_zup=np.asarray(archive["pose_aa_zup"], dtype=np.float64),
                betas=np.asarray(archive["betas"], dtype=np.float64),
                body_scale=float(np.asarray(archive["body_scale"]).item()),
                root_positions_zup=np.asarray(
                    archive["root_positions_zup"], dtype=np.float64
                ),
                transl_zup=np.asarray(archive["transl_zup"], dtype=np.float64),
                source_frame_idx=np.asarray(
                    archive["source_frame_idx"], dtype=np.int64
                ),
                fps=float(np.asarray(archive["fps"]).item()),
                metadata=json.loads(str(np.asarray(archive["metadata_json"]).item())),
            )
        value.validate()
        return value

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".npz", dir=output.parent
        )
        os.close(descriptor)
        try:
            np.savez_compressed(
                temporary,
                pose_aa_zup=self.pose_aa_zup,
                betas=self.betas,
                body_scale=np.asarray(self.body_scale, dtype=np.float64),
                root_positions_zup=self.root_positions_zup,
                transl_zup=self.transl_zup,
                source_frame_idx=self.source_frame_idx,
                fps=np.asarray(self.fps, dtype=np.float64),
                metadata_json=np.asarray(
                    json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _lazy_model(model_root: Path, batch_size: int, device: str):
    try:
        import smplx
        import torch
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError(
            "SMPL fitting requires the project models extra (torch and smplx)"
        ) from error
    model = smplx.create(
        str(model_root),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        use_pca=False,
        batch_size=batch_size,
    )
    return model.to(torch.device(device))


def _axis_angle_to_matrix_torch(axis_angle):
    import torch

    vector = axis_angle
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    axis = vector / torch.clamp(angle, min=1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(
        *vector.shape[:-1], 3, 3
    )
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    return identity + sine * skew + (1.0 - cosine) * (skew @ skew)


def _matrix_to_axis_angle_numpy(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_rotvec()


def fit_lafan_to_smpl_skin(
    repo_root: str | Path = ".",
    *,
    sequence_manifest: str | Path = "manifests/pilot_sequence.yaml",
    output: str | Path | None = None,
    evidence: str | Path | None = None,
    iterations: int = 240,
    device: str = "auto",
) -> dict[str, Any]:
    """Fit neutral-gender SMPL pose/shape to canonical LAFAN joint positions."""

    if iterations < 20:
        raise ValueError("SMPL fitting requires at least 20 iterations")
    root = Path(repo_root).resolve()
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    source_path = root / str(sequence["canonical_path"])
    human = CanonicalHuman.load(source_path)
    source_names = {
        name: index for index, name in enumerate(human.joint_names.astype(str))
    }
    missing = [name for name, _ in SMPL_JOINT_MAP if name not in source_names]
    if missing:
        raise ValueError(f"Canonical LAFAN source lacks SMPL fit joints: {missing}")
    model_root = default_smpl_model_root(root)
    model_file = default_smpl_model_file(root)
    if not model_file.is_file():
        raise FileNotFoundError(f"Chumpy-free neutral SMPL is missing: {model_file}")

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SMPL fitting but is unavailable")
    torch.manual_seed(20260725)
    if device == "cuda":
        torch.cuda.manual_seed_all(20260725)

    frames = len(human.timestamps)
    canonical_to_smpl = SMPL_Y_UP_TO_CANONICAL_Z_UP
    target_native = human.world_positions @ canonical_to_smpl
    source_indices = np.asarray(
        [source_names[name] for name, _ in SMPL_JOINT_MAP], dtype=np.int64
    )
    smpl_indices = np.asarray([index for _, index in SMPL_JOINT_MAP], dtype=np.int64)
    target = torch.as_tensor(
        target_native[:, source_indices], dtype=torch.float32, device=device
    )
    target = target - target[:, :1]
    model = _lazy_model(model_root, frames, device)
    body_pose = torch.nn.Parameter(torch.zeros(frames, 69, device=device))
    global_orient = torch.nn.Parameter(torch.zeros(frames, 3, device=device))
    betas = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(1.02), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        (
            {"params": (body_pose, global_orient), "lr": 0.05},
            {"params": (betas, log_scale), "lr": 0.01},
        )
    )
    joint_indices = torch.as_tensor(smpl_indices, device=device)
    history: list[dict[str, float]] = []
    for iteration in range(iterations + 1):
        optimizer.zero_grad()
        result = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas.expand(frames, -1),
            transl=torch.zeros(frames, 3, device=device),
            return_verts=False,
        )
        predicted = (
            result.joints[:, joint_indices] - result.joints[:, :1]
        ) * log_scale.exp()
        errors = torch.linalg.vector_norm(predicted - target, dim=-1)
        position_loss = torch.mean(errors.square())
        pose_prior = torch.mean(body_pose.square())
        temporal = torch.mean((body_pose[1:] - body_pose[:-1]).square()) + torch.mean(
            (global_orient[1:] - global_orient[:-1]).square()
        )
        shape_prior = torch.mean(betas.square())
        loss = (
            position_loss
            + 1e-4 * pose_prior
            + 2e-3 * temporal
            + 1e-3 * shape_prior
        )
        if iteration < iterations:
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                betas.clamp_(-3.0, 3.0)
                log_scale.clamp_(math.log(0.7), math.log(1.3))
                body_pose.clamp_(-math.pi, math.pi)
        if iteration % 20 == 0 or iteration == iterations:
            history.append(
                {
                    "iteration": float(iteration),
                    "mpjpe_m": float(errors.detach().mean().cpu()),
                    "max_joint_error_m": float(errors.detach().max().cpu()),
                    "loss": float(loss.detach().cpu()),
                }
            )

    native_root_matrix = _axis_angle_to_matrix_torch(global_orient).detach().cpu().numpy()
    zup_root_matrix = np.einsum(
        "ij,tjk->tik", SMPL_Y_UP_TO_CANONICAL_Z_UP, native_root_matrix
    )
    pose_aa_zup = np.concatenate(
        (
            _matrix_to_axis_angle_numpy(zup_root_matrix),
            body_pose.detach().cpu().numpy(),
        ),
        axis=1,
    )
    fitted_betas = betas.detach().cpu().numpy()[0].astype(np.float64)
    fitted_scale = float(log_scale.detach().exp().cpu())
    roots_zup = np.asarray(human.world_positions[:, 0], dtype=np.float64)

    # AMASS-style translation is the additive model translation.  Compute it
    # from the fitted root joint rather than assuming the SMPL origin equals
    # the pelvis joint.
    zup_pose_tensor = torch.as_tensor(
        pose_aa_zup, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        zup_result = model(
            global_orient=zup_pose_tensor[:, :3],
            body_pose=zup_pose_tensor[:, 3:],
            betas=torch.as_tensor(
                fitted_betas, dtype=torch.float32, device=device
            )[None].expand(frames, -1),
            transl=torch.zeros(frames, 3, device=device),
            return_verts=False,
        )
    unshifted_root = zup_result.joints[:, 0].detach().cpu().numpy()
    transl_zup = roots_zup - unshifted_root

    final_errors = errors.detach().cpu().numpy()
    per_joint = {
        name: float(final_errors[:, position].mean())
        for position, (name, _) in enumerate(SMPL_JOINT_MAP)
    }
    metadata = {
        "schema_version": 1,
        "adapter": "lafan_bvh_joint_position_fit_to_smpl",
        "source_representation": "LAFAN1_BVH",
        "dataset_native_smpl": False,
        "body_model": "SMPL neutral, chumpy-free",
        "body_model_file": str(model_file),
        "body_model_sha256": sha256_file(model_file),
        "canonical_source_path": str(source_path.relative_to(root)),
        "canonical_source_file_sha256": sha256_file(source_path),
        "canonical_source_embedded_sha256": human.source_sha256,
        "frames": frames,
        "fps": float(human.fps),
        "fit_joint_count": len(SMPL_JOINT_MAP),
        "fit_iterations": iterations,
        "optimizer": "Adam",
        "device": device,
        "root_aligned_mpjpe_m": float(final_errors.mean()),
        "max_joint_error_m": float(final_errors.max()),
        "per_joint_mean_error_m": per_joint,
        "body_scale": fitted_scale,
        "betas": fitted_betas.tolist(),
        "coordinate_system": "right-handed xyz, +z up",
        "scientific_role": (
            "visualization-only fitted skin; not an original LAFAN SMPL asset "
            "and not a retargeter input unless explicitly declared"
        ),
    }
    motion = SmplSkinMotion(
        pose_aa_zup=pose_aa_zup.astype(np.float64),
        betas=fitted_betas,
        body_scale=fitted_scale,
        root_positions_zup=roots_zup,
        transl_zup=transl_zup.astype(np.float64),
        source_frame_idx=np.arange(frames, dtype=np.int64),
        fps=float(human.fps),
        metadata=metadata,
    )
    output_path = Path(output) if output is not None else default_skin_cache(root)
    if not output_path.is_absolute():
        output_path = root / output_path
    evidence_path = (
        Path(evidence)
        if evidence is not None
        else output_path.with_name("lafan_bvh_fitted_smpl.evidence.json")
    )
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    motion.save(output_path)
    evidence_value = {
        **metadata,
        "output_path": str(output_path.relative_to(root)),
        "output_sha256": sha256_file(output_path),
        "history": history,
    }
    atomic_write_json(evidence_path, evidence_value)
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "evidence": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "root_aligned_mpjpe_m": metadata["root_aligned_mpjpe_m"],
        "max_joint_error_m": metadata["max_joint_error_m"],
        "body_scale": fitted_scale,
        "device": device,
    }


def smpl_mesh_sequence(
    motion: SmplSkinMotion,
    repo_root: str | Path = ".",
    *,
    betas: np.ndarray | None = None,
    body_scale: float | None = None,
    frame_limit: int | None = None,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize fitted SMPL vertices/joints in canonical Z-up coordinates."""

    motion.validate()
    if batch_size < 1:
        raise ValueError("SMPL mesh batch size must be positive")
    frames = len(motion.pose_aa_zup)
    if frame_limit is not None:
        frames = min(frames, frame_limit)
    shape = np.asarray(motion.betas if betas is None else betas, dtype=np.float64)
    if shape.shape != (10,):
        raise ValueError("SMPL mesh override betas must have shape [10]")
    scale = float(motion.body_scale if body_scale is None else body_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("SMPL mesh scale must be finite and positive")

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _lazy_model(default_smpl_model_root(repo_root), batch_size, device)
    vertices: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    for start in range(0, frames, batch_size):
        end = min(frames, start + batch_size)
        pose = torch.as_tensor(
            motion.pose_aa_zup[start:end], dtype=torch.float32, device=device
        )
        roots = torch.as_tensor(
            motion.root_positions_zup[start:end], dtype=torch.float32, device=device
        )
        shape_tensor = torch.as_tensor(
            shape, dtype=torch.float32, device=device
        )[None].expand(end - start, -1)
        with torch.no_grad():
            result = model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                betas=shape_tensor,
                transl=torch.zeros(end - start, 3, device=device),
                return_verts=True,
            )
        pelvis = result.joints[:, 0:1]
        vertices.append(
            ((result.vertices - pelvis) * scale + roots[:, None])
            .detach()
            .cpu()
            .numpy()
        )
        joints.append(
            ((result.joints[:, :24] - pelvis) * scale + roots[:, None])
            .detach()
            .cpu()
            .numpy()
        )
    return (
        np.concatenate(vertices).astype(np.float32),
        np.asarray(model.faces, dtype=np.int32),
        np.concatenate(joints).astype(np.float32),
    )
