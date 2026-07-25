"""Audited PHC fitting and three-way Rerun visualization.

The public PHC G1 fitting configuration is deliberately kept separate from the
Stage 1 canonical G1-29 comparison.  It drives 37 joints (23 body joints plus
14 hand/finger joints) and therefore cannot be silently converted into the
canonical 29-DoF contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, load_yaml, sha256_file
from .smpl_skinning import (
    SmplSkinMotion,
    default_skin_cache,
    default_smpl_model_file,
    default_smpl_model_root,
    smpl_mesh_sequence,
)


PHC_COMMIT = "846988d433ce1f341e85ac6fbd2cd51911bb3341"
PHC_CONFIG = "unitree_g1_fitting"
PHC_ACTUATED_DOFS = 37
PHC_BODY_DOFS = 23
PHC_HAND_DOFS = 14
PHC_VISUAL_MESHES = 43


def _artifact_directory(root: Path, sequence_id: str) -> Path:
    return root / "artifacts" / "phc_visualization" / sequence_id


def _prepared_motion_path(root: Path, sequence_id: str) -> Path:
    return _artifact_directory(root, sequence_id) / "phc_official_g1_37dof.npz"


def _visual_cache_path(root: Path, sequence_id: str) -> Path:
    return _artifact_directory(root, sequence_id) / "phc_g1_visual_transforms.npz"


def _preparation_manifest_path(root: Path) -> Path:
    return root / "manifests" / "phc_visualization_preparation.json"


def _recording_manifest_path(root: Path) -> Path:
    return root / "manifests" / "phc_rerun_visualization.json"


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_head(checkout: Path) -> str:
    process = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Cannot inspect the PHC checkout: {process.stderr.strip()}")
    return process.stdout.strip()


def _ensure_phc_smpl_link(checkout: Path, model_root: Path) -> Path:
    link = checkout / "data" / "smpl"
    link.parent.mkdir(parents=True, exist_ok=True)
    expected = (model_root / "smpl").resolve()
    if link.is_symlink():
        if link.resolve() != expected:
            raise RuntimeError(f"PHC SMPL link points to the wrong model directory: {link}")
    elif link.exists():
        if link.resolve() != expected:
            raise RuntimeError(f"PHC data/smpl exists but is not the audited model: {link}")
    else:
        link.symlink_to(expected, target_is_directory=True)
    return link


def _run_logged(
    command: tuple[str, ...],
    *,
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    environment.setdefault("HYDRA_FULL_ERROR", "1")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Official PHC command failed with exit code {process.returncode}; "
            f"inspect {log_path}"
        )
    return {
        "command": list(command),
        "cwd": str(cwd),
        "elapsed_s": elapsed,
        "exit_code": process.returncode,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


@dataclass(frozen=True)
class PhcPreparedMotion:
    qpos: np.ndarray
    smpl_joints: np.ndarray
    shape_betas: np.ndarray
    body_scale: float
    source_frame_idx: np.ndarray
    fps: float
    metadata: dict[str, Any]

    def validate(self) -> None:
        frames = len(self.qpos)
        if self.qpos.shape != (frames, 7 + PHC_ACTUATED_DOFS):
            raise ValueError("PHC qpos must be free root plus 37 actuated joints")
        if self.smpl_joints.ndim != 3 or self.smpl_joints.shape[:2] != (
            frames,
            24,
        ):
            raise ValueError("PHC scaled SMPL targets must have shape [T,24,3]")
        if self.shape_betas.shape != (10,):
            raise ValueError("PHC fitted SMPL shape must contain 10 coefficients")
        if not np.array_equal(self.source_frame_idx, np.arange(frames)):
            raise ValueError("PHC source-frame indices must be contiguous")
        if (
            not np.isfinite(self.qpos).all()
            or not np.isfinite(self.smpl_joints).all()
            or not np.isfinite(self.shape_betas).all()
            or not np.isfinite(self.body_scale)
            or self.body_scale <= 0.0
            or not np.isfinite(self.fps)
            or self.fps <= 0.0
        ):
            raise ValueError("PHC prepared motion contains non-finite values")
        quaternion_norm = np.linalg.norm(self.qpos[:, 3:7], axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1e-5):
            raise ValueError("PHC root quaternions are not normalized")
        if self.metadata.get("canonical_g1_29_compatible") is not False:
            raise ValueError("PHC 37-DoF output must not claim canonical G1-29 compatibility")

    def save(self, path: Path) -> None:
        self.validate()
        _atomic_savez(
            path,
            qpos=self.qpos,
            smpl_joints=self.smpl_joints,
            shape_betas=self.shape_betas,
            body_scale=np.asarray(self.body_scale, dtype=np.float64),
            source_frame_idx=self.source_frame_idx,
            fps=np.asarray(self.fps, dtype=np.float64),
            metadata_json=np.asarray(
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "PhcPreparedMotion":
        with np.load(path, allow_pickle=False) as archive:
            motion = cls(
                qpos=np.asarray(archive["qpos"], dtype=np.float64),
                smpl_joints=np.asarray(archive["smpl_joints"], dtype=np.float64),
                shape_betas=np.asarray(archive["shape_betas"], dtype=np.float64),
                body_scale=float(np.asarray(archive["body_scale"]).item()),
                source_frame_idx=np.asarray(archive["source_frame_idx"], dtype=np.int64),
                fps=float(np.asarray(archive["fps"]).item()),
                metadata=json.loads(str(np.asarray(archive["metadata_json"]).item())),
            )
        motion.validate()
        return motion


@dataclass(frozen=True)
class PhcRobotVisualCache:
    translations: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    colors: np.ndarray
    mesh_paths: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        frames, visuals = self.translations.shape[:2]
        if self.translations.shape != (frames, visuals, 3):
            raise ValueError("PHC visual translations have the wrong shape")
        if self.rotations.shape != (frames, visuals, 3, 3):
            raise ValueError("PHC visual rotations have the wrong shape")
        if self.scales.shape != (visuals, 3):
            raise ValueError("PHC visual scales have the wrong shape")
        if self.colors.shape != (visuals, 4) or len(self.mesh_paths) != visuals:
            raise ValueError("PHC visual asset inventory is inconsistent")
        if visuals != PHC_VISUAL_MESHES:
            raise ValueError(f"PHC public G1 should expose {PHC_VISUAL_MESHES} visual meshes")
        if not all(
            np.isfinite(value).all()
            for value in (self.translations, self.rotations, self.scales, self.colors)
        ):
            raise ValueError("PHC visual cache contains non-finite values")
        determinant = np.linalg.det(self.rotations.reshape(-1, 3, 3))
        if not np.allclose(determinant, 1.0, atol=1e-5):
            raise ValueError("PHC visual transforms contain invalid rotations")

    def save(self, path: Path) -> None:
        self.validate()
        _atomic_savez(
            path,
            translations=self.translations,
            rotations=self.rotations,
            scales=self.scales,
            colors=self.colors,
            mesh_paths=np.asarray(self.mesh_paths),
            metadata_json=np.asarray(
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "PhcRobotVisualCache":
        with np.load(path, allow_pickle=False) as archive:
            cache = cls(
                translations=np.asarray(archive["translations"], dtype=np.float64),
                rotations=np.asarray(archive["rotations"], dtype=np.float64),
                scales=np.asarray(archive["scales"], dtype=np.float64),
                colors=np.asarray(archive["colors"], dtype=np.float64),
                mesh_paths=tuple(np.asarray(archive["mesh_paths"]).astype(str)),
                metadata=json.loads(str(np.asarray(archive["metadata_json"]).item())),
            )
        cache.validate()
        return cache


def _prepare_robot_visual_cache(
    motion: PhcPreparedMotion,
    robot_xml: Path,
) -> PhcRobotVisualCache:
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError("PHC preparation requires MuJoCo; use conda env 'capture'") from error

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if (model.nq, model.nu) != (44, PHC_ACTUATED_DOFS):
        raise ValueError(
            f"Unexpected PHC robot contract nq={model.nq}, nu={model.nu}"
        )
    fitting_config_path = (
        robot_xml.parents[3] / "cfg" / "robot" / "unitree_g1_fitting.yaml"
    )
    fitting_config = load_yaml(fitting_config_path)
    dof_link_names = tuple(map(str, fitting_config["dof_names"]))
    expected_joint_names = tuple(
        name.removesuffix("_link") + "_joint" for name in dof_link_names
    )
    xml_joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )
    if xml_joint_names != expected_joint_names:
        raise ValueError("PHC fitting DoF order does not match the MuJoCo qpos order")
    data = mujoco.MjData(model)
    visual_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[geom_id] == 0
        and model.geom_conaffinity[geom_id] == 0
    )
    if len(visual_geom_ids) != PHC_VISUAL_MESHES:
        raise ValueError("The PHC XML visual-geometry inventory is stale")

    mesh_file_by_name = {
        element.attrib["name"]: element.attrib["file"]
        for element in ET.parse(robot_xml).getroot().findall("./asset/mesh")
    }
    mesh_directory = robot_xml.parent / "meshes"
    mesh_paths: list[str] = []
    scales = np.empty((len(visual_geom_ids), 3), dtype=np.float64)
    colors = np.empty((len(visual_geom_ids), 4), dtype=np.float64)
    for position, geom_id in enumerate(visual_geom_ids):
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        path = (mesh_directory / mesh_file_by_name[mesh_name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PHC robot mesh is missing: {path}")
        mesh_paths.append(str(path))
        scales[position] = model.mesh_scale[mesh_id]
        colors[position] = model.geom_rgba[geom_id]

    translations = np.empty(
        (len(motion.qpos), len(visual_geom_ids), 3), dtype=np.float64
    )
    rotations = np.empty(
        (len(motion.qpos), len(visual_geom_ids), 3, 3), dtype=np.float64
    )
    for frame, qpos in enumerate(motion.qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        translations[frame] = data.geom_xpos[list(visual_geom_ids)]
        rotations[frame] = data.geom_xmat[list(visual_geom_ids)].reshape(-1, 3, 3)

    metadata = {
        "schema_version": 1,
        "robot_xml": str(robot_xml),
        "robot_xml_sha256": sha256_file(robot_xml),
        "nq": model.nq,
        "nv": model.nv,
        "actuated_dofs": model.nu,
        "dof_link_names": list(dof_link_names),
        "qpos_joint_names": list(xml_joint_names),
        "visual_mesh_count": len(visual_geom_ids),
        "visual_mesh_sha256": {
            path: sha256_file(path) for path in mesh_paths
        },
    }
    return PhcRobotVisualCache(
        translations=translations,
        rotations=rotations,
        scales=scales,
        colors=colors,
        mesh_paths=tuple(mesh_paths),
        metadata=metadata,
    )


def prepare_phc_visualization(
    repo_root: str | Path = ".",
    *,
    sequence_manifest: str | Path = "manifests/pilot_sequence.yaml",
    fitting_iterations: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    """Run the frozen public PHC SMPL→G1 fitting path and cache clean evidence."""

    if fitting_iterations != 500:
        raise ValueError(
            "The scientific PHC visualization freezes the official 500 motion-fit iterations"
        )
    root = Path(repo_root).resolve()
    sequence_path = Path(sequence_manifest)
    if not sequence_path.is_absolute():
        sequence_path = root / sequence_path
    sequence = load_yaml(sequence_path)
    sequence_id = str(sequence["sequence_id"])
    skin_path = default_skin_cache(root)
    if not skin_path.is_file():
        raise FileNotFoundError("Run `rtcmp prepare-smpl-skin` before PHC fitting")
    skin = SmplSkinMotion.load(skin_path)

    checkout = root / "external" / "PHC"
    if _git_head(checkout) != PHC_COMMIT:
        raise RuntimeError("The PHC checkout does not match the frozen official commit")
    model_root = default_smpl_model_root(root)
    model_file = default_smpl_model_file(root)
    smpl_link = _ensure_phc_smpl_link(checkout, model_root)
    artifact_dir = _artifact_directory(root, sequence_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_dir = artifact_dir / "amass_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    amass_input = input_dir / "lafan_bvh_fitted_smpl.npz"

    # PHC's official loader keeps only the first 66 SMPL pose components and
    # zeroes the final six.  Save the complete fitted pose here and let the
    # unmodified upstream loader perform that policy.
    _atomic_savez(
        amass_input,
        poses=skin.pose_aa_zup.astype(np.float64),
        trans=skin.transl_zup.astype(np.float64),
        betas=skin.betas.astype(np.float64),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(skin.fps, dtype=np.float64),
    )

    official_shape = checkout / "data" / "g1" / "shape_optimized_v1.pkl"
    key = f"0-{amass_input.stem}"
    official_motion = (
        checkout / "data" / "g1" / "v1" / "singles" / f"{key}.pkl"
    )
    repository_root = Path(repo_root).resolve()
    previous_manifest_path = _preparation_manifest_path(repository_root)
    previous_manifest: dict[str, Any] = {}
    if previous_manifest_path.is_file():
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
    previous_phc = previous_manifest.get("phc", {})
    previous_smpl = previous_manifest.get("smpl_adapter", {})
    previous_amass = previous_manifest.get("amass_input", {})
    shape_reusable = bool(
        official_shape.is_file()
        and previous_phc.get("commit") == PHC_COMMIT
        and previous_phc.get("config") == PHC_CONFIG
        and previous_smpl.get("body_model_sha256") == sha256_file(model_file)
        and previous_phc.get("official_shape_sha256")
        == sha256_file(official_shape)
    )
    motion_reusable = bool(
        official_motion.is_file()
        and shape_reusable
        and int(previous_phc.get("fitting_iterations", -1))
        == fitting_iterations
        and previous_amass.get("sha256") == sha256_file(amass_input)
        and previous_phc.get("official_motion_sha256")
        == sha256_file(official_motion)
    )
    command_records: list[dict[str, Any]] = []
    shape_ran = force or not shape_reusable
    if shape_ran:
        command_records.append(
            _run_logged(
                (
                    sys.executable,
                    "scripts/data_process/fit_smpl_shape.py",
                    f"robot={PHC_CONFIG}",
                ),
                cwd=checkout,
                log_path=artifact_dir / "fit_smpl_shape.log",
            )
        )
    if force or shape_ran or not motion_reusable:
        command_records.append(
            _run_logged(
                (
                    sys.executable,
                    "scripts/data_process/fit_smpl_motion.py",
                    f"robot={PHC_CONFIG}",
                    f"+amass_root={input_dir}",
                    "+fit_all=True",
                    f"+fitting_iterations={fitting_iterations}",
                ),
                cwd=checkout,
                log_path=artifact_dir / "fit_smpl_motion.log",
            )
        )
    if not official_shape.is_file() or not official_motion.is_file():
        raise RuntimeError("Official PHC fitting did not create its declared outputs")

    try:
        import joblib
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError("PHC preparation requires joblib; use conda env 'capture'") from error

    shape_tensor, scale_tensor = joblib.load(official_shape)
    shape = np.asarray(shape_tensor.detach().cpu(), dtype=np.float64).reshape(-1)
    scale = float(np.asarray(scale_tensor.detach().cpu()).reshape(-1)[0])
    raw_bundle = joblib.load(official_motion)
    if set(raw_bundle) != {key}:
        raise ValueError(f"Official PHC output keys are stale: {sorted(raw_bundle)}")
    raw = raw_bundle[key]
    root_translation = np.asarray(
        raw["root_trans_offset"], dtype=np.float64
    ).reshape(-1, 3)
    root_xyzw = np.asarray(raw["root_rot"], dtype=np.float64).reshape(-1, 4)
    dof = np.asarray(raw["dof"], dtype=np.float64).reshape(-1, PHC_ACTUATED_DOFS)
    frames = len(root_translation)
    if frames != len(skin.pose_aa_zup):
        raise ValueError("Official PHC output does not cover the full frozen source")
    qpos = np.concatenate(
        (root_translation, root_xyzw[:, [3, 0, 1, 2]], dof),
        axis=1,
    )
    metadata = {
        "schema_version": 1,
        "method": "PHC public SMPL-to-G1 fitting utility",
        "scientific_role": "separate visualization; not a canonical Stage 1 operating point",
        "phc_commit": PHC_COMMIT,
        "phc_config": PHC_CONFIG,
        "fitting_iterations": fitting_iterations,
        "source_representation": "LAFAN1_BVH_fitted_to_SMPL",
        "source_skin_sha256": sha256_file(skin_path),
        "amass_adapter_sha256": sha256_file(amass_input),
        "actor_betas_ignored_by_official_motion_fit": True,
        "official_neutral_robot_fitted_shape_used": True,
        "actuated_dofs": PHC_ACTUATED_DOFS,
        "body_dofs": PHC_BODY_DOFS,
        "hand_finger_dofs": PHC_HAND_DOFS,
        "canonical_g1_29_compatible": False,
        "root_quaternion_convention": "wxyz",
    }
    prepared = PhcPreparedMotion(
        qpos=qpos,
        smpl_joints=np.asarray(raw["smpl_joints"], dtype=np.float64),
        shape_betas=shape,
        body_scale=scale,
        source_frame_idx=np.arange(frames, dtype=np.int64),
        fps=float(raw["fps"]),
        metadata=metadata,
    )
    prepared_path = _prepared_motion_path(Path(repo_root).resolve(), sequence_id)
    prepared.save(prepared_path)

    robot_xml = (
        checkout / "phc" / "data" / "assets" / "robot" / "unitree_g1" / "g1.xml"
    ).resolve()
    visual_cache = _prepare_robot_visual_cache(prepared, robot_xml)
    visual_path = _visual_cache_path(Path(repo_root).resolve(), sequence_id)
    visual_cache.save(visual_path)

    prior_commands = list(
        previous_manifest.get("commands_executed_this_run", ())
    )
    current_scripts = {
        str(record.get("command", ("", ""))[1])
        for record in command_records
        if len(record.get("command", ())) > 1
    }
    command_records = [
        record
        for record in prior_commands
        if len(record.get("command", ())) > 1
        and str(record["command"][1]) not in current_scripts
    ] + command_records

    def portable_path(path: Path) -> str:
        try:
            return path.relative_to(repository_root).as_posix()
        except ValueError:
            return str(path)

    portable_commands: list[dict[str, Any]] = []
    for record in command_records:
        portable = dict(record)
        command = [
            str(value).replace(str(repository_root), ".")
            for value in portable.get("command", ())
        ]
        portable["command"] = command
        portable["cwd"] = portable_path(Path(str(portable["cwd"])))
        portable["log_path"] = portable_path(Path(str(portable["log_path"])))
        portable_commands.append(portable)
    manifest = {
        "schema_version": 1,
        "sequence_id": sequence_id,
        "source_format": "LAFAN1 22-joint BVH",
        "dataset_native_smpl": False,
        "smpl_adapter": {
            "path": skin_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(skin_path),
            "body_model_sha256": sha256_file(model_file),
            "fit_root_aligned_mpjpe_m": skin.metadata["root_aligned_mpjpe_m"],
        },
        "amass_input": {
            "path": amass_input.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(amass_input),
        },
        "phc": {
            "commit": PHC_COMMIT,
            "config": PHC_CONFIG,
            "fitting_iterations": fitting_iterations,
            "smpl_link": portable_path(smpl_link),
            "official_shape_path": portable_path(official_shape),
            "official_shape_sha256": sha256_file(official_shape),
            "official_motion_path": portable_path(official_motion),
            "official_motion_sha256": sha256_file(official_motion),
            "actuated_dofs": PHC_ACTUATED_DOFS,
            "canonical_g1_29_compatible": False,
            "shape_betas": shape.tolist(),
            "body_scale": scale,
        },
        "prepared_motion": {
            "path": prepared_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(prepared_path),
            "frames": frames,
        },
        "robot_visual_cache": {
            "path": visual_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(visual_path),
            "mesh_count": PHC_VISUAL_MESHES,
        },
        "commands_executed_this_run": portable_commands,
    }
    atomic_write_json(_preparation_manifest_path(repository_root), manifest)
    return manifest


def _rotation_matrices_from_pose(pose_aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(pose_aa[:, :3]).as_matrix()


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = np.asarray(quaternion)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix()


def _root_local_vertices(
    vertices: np.ndarray, roots: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    return np.einsum("tvi,tij->tvj", vertices - roots[:, None, :], rotations)


def _root_local_robot(
    translations: np.ndarray,
    rotations: np.ndarray,
    roots: np.ndarray,
    root_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_translations = np.einsum(
        "tvi,tij->tvj", translations - roots[:, None, :], root_rotations
    )
    local_rotations = np.einsum(
        "tji,tvjk->tvik", root_rotations, rotations
    )
    return local_translations, local_rotations


def _log_mesh_topology(rr: Any, entity: str, faces: np.ndarray, color: tuple[int, ...]) -> None:
    rr.log(
        entity,
        rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=color),
        static=True,
    )


def write_phc_rerun_visualization(
    repo_root: str | Path = ".",
    *,
    output: str | Path = "artifacts/visualization/phc_scale_three_way.rrd",
    manifest: str | Path = "manifests/phc_rerun_visualization.json",
    spawn: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Visualize original-size SMPL, PHC-scaled SMPL, and PHC's 37-DoF G1."""

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:  # pragma: no cover - environment error path
        raise RuntimeError("Rerun >=0.34 is required; use conda env 'vis'") from error
    version = tuple(int(part) for part in rr.__version__.split(".")[:2])
    if version < (0, 34):
        raise RuntimeError("PHC visualization requires Rerun >=0.34")

    root = Path(repo_root).resolve()
    sequence = load_yaml(root / "manifests" / "pilot_sequence.yaml")
    sequence_id = str(sequence["sequence_id"])
    skin_path = default_skin_cache(root)
    prepared_path = _prepared_motion_path(root, sequence_id)
    visual_path = _visual_cache_path(root, sequence_id)
    for path in (skin_path, prepared_path, visual_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"PHC visualization input is missing: {path}. "
                "Run `rtcmp prepare-phc-visualization` in conda env capture."
            )
    skin = SmplSkinMotion.load(skin_path)
    prepared = PhcPreparedMotion.load(prepared_path)
    robot = PhcRobotVisualCache.load(visual_path)
    frames = min(len(prepared.qpos), len(skin.pose_aa_zup))
    if max_frames is not None:
        frames = min(frames, max_frames)
    if frames < 1:
        raise ValueError("PHC visualization has no frames")

    # Match the exact public PHC loader: poses[:, :66] plus six zeros.
    phc_pose = skin.pose_aa_zup.copy()
    phc_pose[:, 66:] = 0.0
    input_skin = replace(skin, pose_aa_zup=phc_pose)
    original_vertices, faces, _ = smpl_mesh_sequence(
        input_skin, root, frame_limit=frames, batch_size=64, device="cpu"
    )
    scaled_vertices, scaled_faces, _ = smpl_mesh_sequence(
        input_skin,
        root,
        betas=prepared.shape_betas,
        body_scale=prepared.body_scale,
        frame_limit=frames,
        batch_size=64,
        device="cpu",
    )
    if not np.array_equal(faces, scaled_faces):
        raise ValueError("SMPL topology changed between original and PHC-scaled bodies")
    original_roots = skin.root_positions_zup[:frames]
    scaled_roots = prepared.smpl_joints[:frames, 0]
    scaled_vertices += (
        scaled_roots - skin.root_positions_zup[:frames]
    )[:, None, :]

    human_root_rotation = _rotation_matrices_from_pose(phc_pose[:frames])
    robot_root_rotation = _quaternion_wxyz_to_matrix(prepared.qpos[:frames, 3:7])
    original_local = _root_local_vertices(
        original_vertices, original_roots, human_root_rotation
    )
    scaled_local = _root_local_vertices(
        scaled_vertices, scaled_roots, human_root_rotation
    )
    robot_local_translation, robot_local_rotation = _root_local_robot(
        robot.translations[:frames],
        robot.rotations[:frames],
        prepared.qpos[:frames, :3],
        robot_root_rotation,
    )

    original_world = original_vertices - original_roots[0, None, :]
    scaled_world = scaled_vertices - scaled_roots[0, None, :]
    robot_world_translation = (
        robot.translations[:frames] - prepared.qpos[0, None, :3]
    )
    original_ground = float(np.min(original_world[0, :, 2]))
    scaled_ground = float(np.min(scaled_world[0, :, 2]))
    robot_ground = float(np.min(robot_world_translation[0, :, 2]))
    original_world[..., 2] -= original_ground
    scaled_world[..., 2] -= scaled_ground
    robot_world_translation[..., 2] -= robot_ground
    lanes = {
        "original": np.asarray((-2.7, 0.0, 0.0)),
        "scaled": np.asarray((0.0, 0.0, 0.0)),
        "robot": np.asarray((2.7, 0.0, 0.0)),
    }

    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                origin="/side_by_side",
                name="Original size · PHC-scaled · PHC G1",
                line_grid=True,
                eye_controls=rrb.EyeControls3D(
                    position=(0.0, -10.0, 3.2),
                    look_target=(0.0, 0.0, 0.85),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            rrb.Spatial3DView(
                origin="/root_overlay",
                name="Root-frame scale and pose overlay",
                line_grid=True,
                eye_controls=rrb.EyeControls3D(
                    position=(2.8, -4.6, 2.2),
                    look_target=(0.0, 0.0, 0.7),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            name="PHC scale narrative",
        ),
        rrb.TimePanel(expanded=True, timeline="time", fps=prepared.fps),
        rrb.SelectionPanel(expanded=False),
        collapse_panels=False,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(prepared_path) + sha256_file(skin_path)
    recording_id = str(uuid.UUID(bytes=bytes.fromhex(digest[:32])))
    rr.init(
        "phc_lafan_scale_three_way",
        recording_id=recording_id,
        strict=True,
        default_blueprint=blueprint,
    )
    rr.save(output_path, default_blueprint=blueprint)
    if spawn:
        rr.spawn(default_blueprint=blueprint)
    rr.send_blueprint(blueprint)
    for view in ("side_by_side", "root_overlay"):
        rr.log(view, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        _log_mesh_topology(
            rr, f"{view}/original_size_smpl", faces, (80, 160, 235, 210)
        )
        _log_mesh_topology(
            rr, f"{view}/phc_scaled_smpl", faces, (236, 156, 69, 185)
        )
        for index, mesh_path in enumerate(robot.mesh_paths):
            color = tuple(
                int(np.clip(channel * 255.0, 0.0, 255.0))
                for channel in robot.colors[index]
            )
            rr.log(
                f"{view}/phc_g1_meshes/{index:02d}_{Path(mesh_path).stem}",
                rr.Asset3D(path=mesh_path, albedo_factor=color),
                static=True,
            )
    readme = "\n".join(
        (
            "# PHC public G1 fitting · scale narrative",
            "",
            "- Original LAFAN1 asset: 22-joint BVH, not SMPL or SMPL-X.",
            "- Blue: visualization-only SMPL fit at the fitted actor size.",
            "- Orange: the same pose after PHC replaces actor shape with its neutral, robot-fitted SMPL shape and scalar.",
            "- Gray: the official public PHC G1 fitting result.",
            f"- PHC shape scalar: {prepared.body_scale:.6f}.",
            f"- PHC embodiment: {PHC_ACTUATED_DOFS} motors ({PHC_BODY_DOFS} body + {PHC_HAND_DOFS} hand/finger), not canonical G1-29.",
            "- This recording is an explanatory PHC visualization, not a Stage 1 metric operating point.",
        )
    )
    rr.log(
        "metadata/readme",
        rr.TextDocument(readme, media_type="text/markdown"),
        static=True,
    )

    for frame in range(frames):
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=float(frame / prepared.fps))
        rr.log("metadata/frame_marker", rr.Scalars(float(frame)))
        rr.log(
            "side_by_side/original_size_smpl",
            rr.Mesh3D.from_fields(
                vertex_positions=original_world[frame] + lanes["original"]
            ),
        )
        rr.log(
            "side_by_side/phc_scaled_smpl",
            rr.Mesh3D.from_fields(
                vertex_positions=scaled_world[frame] + lanes["scaled"]
            ),
        )
        rr.log(
            "root_overlay/original_size_smpl",
            rr.Mesh3D.from_fields(vertex_positions=original_local[frame]),
        )
        rr.log(
            "root_overlay/phc_scaled_smpl",
            rr.Mesh3D.from_fields(vertex_positions=scaled_local[frame]),
        )
        rr.log(
            "side_by_side/labels",
            rr.Points3D(
                [
                    lanes["original"] + (0.0, 0.0, 1.2),
                    lanes["scaled"] + (0.0, 0.0, 1.2),
                    lanes["robot"] + (0.0, 0.0, 1.2),
                ],
                colors=[(80, 160, 235), (236, 156, 69), (210, 210, 210)],
                radii=0.001,
                labels=[
                    "Original-size fitted SMPL",
                    "PHC robot-fitted shape + scale",
                    "PHC public G1 · 37 motors",
                ],
                show_labels=True,
            ),
        )
        for index, mesh_path in enumerate(robot.mesh_paths):
            entity = f"{index:02d}_{Path(mesh_path).stem}"
            rr.log(
                f"side_by_side/phc_g1_meshes/{entity}",
                rr.InstancePoses3D(
                    translations=[
                        robot_world_translation[frame, index] + lanes["robot"]
                    ],
                    mat3x3=[robot.rotations[frame, index]],
                    scales=[robot.scales[index]],
                ),
            )
            rr.log(
                f"root_overlay/phc_g1_meshes/{entity}",
                rr.InstancePoses3D(
                    translations=[robot_local_translation[frame, index]],
                    mat3x3=[robot_local_rotation[frame, index]],
                    scales=[robot.scales[index]],
                ),
            )
    rr.disconnect()

    verify = subprocess.run(
        (
            str(Path.home() / "anaconda3" / "envs" / "vis" / "bin" / "rerun"),
            "rrd",
            "verify",
            "--check-footers",
            "true",
            str(output_path),
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RERUN_ANALYTICS": "disabled"},
    )
    if verify.returncode != 0:
        raise RuntimeError(
            "Rerun rejected the PHC recording: "
            + (verify.stderr.strip() or verify.stdout.strip())
        )
    try:
        portable_output = output_path.relative_to(root).as_posix()
    except ValueError:
        portable_output = str(output_path)
    result = {
        "schema_version": 1,
        "sequence_id": sequence_id,
        "frames": frames,
        "fps": prepared.fps,
        "source_format": "LAFAN1 22-joint BVH",
        "dataset_native_smpl": False,
        "human_rendering": "visualization_only_fitted_smpl",
        "views": ["side_by_side", "root_overlay"],
        "phc": {
            "commit": PHC_COMMIT,
            "config": PHC_CONFIG,
            "actuated_dofs": PHC_ACTUATED_DOFS,
            "body_dofs": PHC_BODY_DOFS,
            "hand_finger_dofs": PHC_HAND_DOFS,
            "canonical_g1_29_compatible": False,
            "body_scale": prepared.body_scale,
            "visual_meshes": PHC_VISUAL_MESHES,
        },
        "inputs": {
            "smpl_skin": {
                "path": skin_path.relative_to(root).as_posix(),
                "sha256": sha256_file(skin_path),
            },
            "prepared_motion": {
                "path": prepared_path.relative_to(root).as_posix(),
                "sha256": sha256_file(prepared_path),
            },
            "robot_visual_cache": {
                "path": visual_path.relative_to(root).as_posix(),
                "sha256": sha256_file(visual_path),
            },
        },
        "output": portable_output,
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "rerun_version": rr.__version__,
        "rrd_verify": "passed",
    }
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    atomic_write_json(manifest_path, result)
    return result
