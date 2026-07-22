"""Official Holosoma two-case Full/No-Hard interaction experiment."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, sha256_file
from .runner import _aggregate_sha256, _conda_python, _git, utc_now
from .schemas import RunManifest, RunStatus


INTERACTION_CASES = {
    "box": {
        "task_name": "sub3_largebox_003",
        "task_type": "object_interaction",
        "data_format": "smplh",
        "relative_data_path": "demo_data/OMOMO_new",
    },
    "climb": {
        "task_name": "mocap_climb_seq_0",
        "task_type": "climbing",
        "data_format": "mocap",
        "relative_data_path": "demo_data/climb",
    },
}


def mesh_surface_diagnostics(
    model: Any,
    qpos: np.ndarray,
    object_token: str,
    foot_body_names: list[str],
    foot_sticking: list[dict[str, bool]],
    foot_tolerance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute distances with MuJoCo's actual geometry surface query."""
    import mujoco

    data = mujoco.MjData(model)
    object_geoms = []
    robot_geoms = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if object_token in name or object_token in body_name:
            object_geoms.append(geom_id)
        elif body_id != 0 and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            robot_geoms.append(geom_id)
    if not object_geoms or not robot_geoms:
        raise RuntimeError("Could not identify both robot and object collision surfaces")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names
    ]
    if any(value < 0 for value in foot_ids):
        raise RuntimeError("Interaction scene is missing a configured foot-sticking body")
    rows: list[dict[str, Any]] = []
    previous_feet: np.ndarray | None = None
    for frame, value in enumerate(qpos):
        data.qpos[:] = value
        mujoco.mj_forward(model, data)
        fromto = np.zeros(6, dtype=np.float64)
        distances = [
            float(mujoco.mj_geomDistance(model, data, robot, obj, 10.0, fromto))
            for robot in robot_geoms
            for obj in object_geoms
        ]
        distance = min(distances)
        feet = data.xpos[foot_ids].copy()
        displacement = np.zeros(len(feet)) if previous_feet is None else np.linalg.norm(
            feet[:, :2] - previous_feet[:, :2], axis=1
        )
        previous_feet = feet
        flags = foot_sticking[frame] if frame < len(foot_sticking) else {}
        left_stance = any(bool(flag) for key, flag in flags.items() if key.lower().startswith("l"))
        right_stance = any(bool(flag) for key, flag in flags.items() if key.lower().startswith("r"))
        left_motion = float(max((d for d, name in zip(displacement, foot_body_names) if "left" in name), default=0.0))
        right_motion = float(max((d for d, name in zip(displacement, foot_body_names) if "right" in name), default=0.0))
        rows.append(
            {
                "frame": frame,
                "minimum_robot_object_surface_distance_m": distance,
                "strict_contact_2cm": distance <= 0.02,
                "near_contact_5cm": distance <= 0.05,
                "proximity_10cm": distance <= 0.10,
                "penetration_depth_m": max(0.0, -distance),
                "left_stance": left_stance,
                "right_stance": right_stance,
                "left_foot_xy_displacement_m": left_motion,
                "right_foot_xy_displacement_m": right_motion,
                "foot_sticking_violation": (
                    (left_stance and left_motion > np.sqrt(2.0) * foot_tolerance_m + 1e-4)
                    or (right_stance and right_motion > np.sqrt(2.0) * foot_tolerance_m + 1e-4)
                ),
            }
        )
    distances = np.asarray([row["minimum_robot_object_surface_distance_m"] for row in rows])
    summary = {
        "frames": len(rows),
        "robot_collision_geom_count": len(robot_geoms),
        "object_collision_geom_count": len(object_geoms),
        "distance_backend": "mujoco.mj_geomDistance over collision surfaces",
        "strict_contact_2cm_frame_rate": float(np.mean(distances <= 0.02)),
        "near_contact_5cm_frame_rate": float(np.mean(distances <= 0.05)),
        "proximity_10cm_frame_rate": float(np.mean(distances <= 0.10)),
        "penetration_any_frame_rate": float(np.mean(distances < 0.0)),
        "penetration_frame_rate": float(np.mean(distances < -0.0011)),
        "penetration_primary_threshold_m": 0.0011,
        "maximum_penetration_depth_m": float(max(0.0, -distances.min())),
        "foot_sticking_violation_frame_rate": float(
            np.mean([row["foot_sticking_violation"] for row in rows])
        ),
    }
    return rows, summary


def run_interaction_native(
    repo_root: str | Path,
    case: str,
    variant: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    if case not in INTERACTION_CASES or variant not in {"full", "no-hard"}:
        raise ValueError("Unknown interaction case or variant")
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkout = root / "external" / "holosoma"
    package_root = checkout / "src" / "holosoma_retargeting"
    module_root = str(package_root)
    sys.path.insert(0, module_root)
    original_cwd = Path.cwd()
    try:
        import mujoco

        from holosoma_retargeting.config_types.retargeter import RetargeterConfig
        from holosoma_retargeting.config_types.retargeting import RetargetingConfig
        from holosoma_retargeting.config_types.robot import RobotConfig
        from holosoma_retargeting.examples import robot_retarget
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

        spec = INTERACTION_CASES[case]
        data_path = package_root / "holosoma_retargeting" / spec["relative_data_path"]
        native_output = output / "native_output"
        native_output.mkdir(parents=True, exist_ok=True)
        instances: list[Any] = []
        frame_times: list[float] = []
        steady_times: list[float] = []

        class CapturingRetargeter(InteractionMeshRetargeter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._rtcmp_foot_sticking: list[dict[str, bool]] = []
                instances.append(self)

            def iterate(self, *args, **kwargs):
                start = time.perf_counter()
                result = super().iterate(*args, **kwargs)
                frame_times.append(time.perf_counter() - start)
                return result

            def retarget_motion(self, *args, **kwargs):
                self._rtcmp_foot_sticking = kwargs["foot_sticking_sequences"]
                start = time.perf_counter()
                result = super().retarget_motion(*args, **kwargs)
                steady_times.append(time.perf_counter() - start)
                return result

        original_symbol = robot_retarget.InteractionMeshRetargeter
        robot_retarget.InteractionMeshRetargeter = CapturingRetargeter
        enabled = variant == "full"
        retargeter_cfg = RetargeterConfig(
            activate_obj_non_penetration=enabled,
            activate_foot_sticking=enabled,
            activate_joint_limits=True,
        )
        robot_cfg = RobotConfig(
            robot_type="g1",
            robot_urdf_file=(
                "models/g1/g1_29dof_spherehand.urdf" if case == "climb" else None
            ),
        )
        cfg = RetargetingConfig(
            task_type=spec["task_type"],
            robot="g1",
            data_format=spec["data_format"],
            task_name=spec["task_name"],
            data_path=data_path,
            save_dir=native_output,
            robot_config=robot_cfg,
            retargeter=retargeter_cfg,
        )
        np.random.seed(0)
        os.chdir(package_root / "holosoma_retargeting")
        start = time.perf_counter()
        try:
            robot_retarget.main(cfg)
        finally:
            robot_retarget.InteractionMeshRetargeter = original_symbol
        wall_time = time.perf_counter() - start
        if len(instances) != 1:
            raise RuntimeError("Expected exactly one captured Holosoma retargeter")
        instance = instances[0]
        result_path = native_output / f"{spec['task_name']}_original.npz"
        with np.load(result_path, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            fps = float(data["fps"])
        if len(frame_times) != len(qpos):
            raise RuntimeError("Interaction timing count does not match output frames")
        if len(steady_times) != 1:
            raise RuntimeError("Interaction timing adapter did not observe one sequence loop")
        scene_path = output / "expanded_scene.xml"
        mujoco.mj_saveLastXML(str(scene_path), instance.robot_model)
        rows, metrics = mesh_surface_diagnostics(
            instance.robot_model,
            qpos,
            "largebox" if case == "box" else "multi_boxes",
            list(instance.foot_links),
            instance._rtcmp_foot_sticking,
            instance.foot_sticking_tolerance,
        )
        csv_path = output / "per_frame_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        input_path = (
            data_path / f"{spec['task_name']}.pt"
            if case == "box"
            else next((data_path / spec["task_name"]).glob("*.npy"))
        )
        mesh_paths = (
            [package_root / "holosoma_retargeting" / "models" / "largebox" / "largebox.obj"]
            if case == "box"
            else sorted((data_path / spec["task_name"] / "box_models").glob("*.obj"))
        )
        metrics.update(
            {
                "case": case,
                "task_name": spec["task_name"],
                "variant": variant,
                "status": "succeeded",
                "fps": fps,
                "wall_time_s": wall_time,
                "initialization_and_preprocess_time_s": max(0.0, wall_time - steady_times[0]),
                "steady_end_to_end_time_s": steady_times[0],
                "native_frame_times_s": frame_times,
                "native_median_frame_s": float(np.median(frame_times)),
                "native_core_rtf": float(sum(frame_times) / (len(qpos) / fps)),
                "end_to_end_rtf": float(steady_times[0] / (len(qpos) / fps)),
                "activate_obj_non_penetration": enabled,
                "activate_foot_sticking": enabled,
                "activate_joint_limits": True,
                "input_sha256": sha256_file(input_path),
                "object_mesh_sha256": _aggregate_sha256(mesh_paths),
                "expanded_scene_sha256": sha256_file(scene_path),
                "qpos_sha256": sha256_file(result_path),
                "qpos_width": int(qpos.shape[1]),
                "upstream_commit": "5f48635a3624656a5f46a07df26d43187e59f855",
                "patch": "patches/holosoma/interaction-hard-constraint-flags.patch",
                "claim_scope": "two-case case-study evidence only",
            }
        )
        atomic_write_json(output / "summary.json", metrics)
        return metrics
    finally:
        os.chdir(original_cwd)
        if sys.path[0] == module_root:
            sys.path.pop(0)


def run_interaction(
    case: str, variant: str, repo_root: str | Path = "."
) -> RunManifest:
    root = Path(repo_root).resolve()
    run_id = f"interaction__{case}__{variant}"
    run_dir = root / "runs" / "interaction" / case / variant
    manifest_path = root / "runs" / "interaction_manifests" / f"{run_id}.json"
    summary_path = run_dir / "summary.json"
    if manifest_path.exists():
        existing = RunManifest.load(manifest_path)
        if existing.status == RunStatus.SUCCEEDED and summary_path.is_file():
            return existing
        attempt = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%SZ")
        archive = manifest_path.with_name(f"{manifest_path.stem}__{attempt}.json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archive)
        run_dir = run_dir / attempt
        summary_path = run_dir / "summary.json"
    python = _conda_python("hsretargeting")
    command = [
        str(python),
        "-m",
        "retargeting_comparison.interaction_worker",
        "--repo-root",
        str(root),
        "--case",
        case,
        "--variant",
        variant,
        "--output-dir",
        str(run_dir),
    ]
    config_hash = _aggregate_sha256(
        [
            root / "patches" / "holosoma" / "interaction-hard-constraint-flags.patch",
            root
            / "external"
            / "holosoma"
            / "src"
            / "holosoma_retargeting"
            / "holosoma_retargeting"
            / "config_types"
            / "retargeter.py",
        ]
    )
    logs = run_dir / "logs"
    manifest = RunManifest(
        run_id=run_id,
        method=f"interaction-{case}-{variant}",
        status=RunStatus.RUNNING,
        command=command,
        environment="conda:hsretargeting",
        repo_commit=_git(root, "rev-parse", "HEAD"),
        config_sha256=config_hash,
        device="CPU; threads=1",
        started_at=utc_now(),
        stdout_log=str(logs / "stdout.log"),
        stderr_log=str(logs / "stderr.log"),
        output_path=str(summary_path),
    )
    manifest.save(manifest_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    logs.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (
        logs / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
    manifest.wall_time_s = time.perf_counter() - start
    manifest.finished_at = utc_now()
    manifest.exit_code = result.returncode
    if result.returncode == 0 and summary_path.is_file():
        manifest.status = RunStatus.SUCCEEDED
        manifest.output_sha256 = sha256_file(summary_path)
    else:
        manifest.status = RunStatus.FAILED
        manifest.message = "Interaction worker failed; inspect stderr log"
    manifest.save(manifest_path)
    return manifest
