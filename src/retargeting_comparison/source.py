"""Deterministic, source-only LAFAN Pilot selection and canonicalization."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bvh import forward_kinematics, parse_bvh
from .io_utils import atomic_write_yaml, sha256_file
from .rotations import quaternion_wxyz_to_matrix, yaw_from_matrix
from .schemas import CanonicalHuman

LAFAN_TO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class SourceFeatures:
    sequence_id: str
    source_file: str
    frame_start: int
    frame_end: int
    frames: int
    fps: float
    duration_s: float
    root_path_m: float
    root_mean_speed_mps: float
    root_yaw_range_rad: float
    wrist_range_m: float
    contact_transitions: int
    eligible: bool
    rejection_reason: str


def _find_joint(names: tuple[str, ...], aliases: tuple[str, ...]) -> int:
    lowered = {name.lower(): index for index, name in enumerate(names)}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    raise ValueError(f"None of {aliases!r} found in source joints")


def _remove_short_true_runs(mask: np.ndarray, minimum: int = 3) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    start = 0
    while start < len(result):
        value = result[start]
        end = start + 1
        while end < len(result) and result[end] == value:
            end += 1
        if value and end - start < minimum:
            result[start:end] = False
        start = end
    return result


def foot_contacts(
    positions: np.ndarray, names: tuple[str, ...], fps: float
) -> np.ndarray:
    left = _find_joint(names, ("LeftFoot", "LeftToe", "LeftToeBase", "L_Ankle"))
    right = _find_joint(names, ("RightFoot", "RightToe", "RightToeBase", "R_Ankle"))
    feet = positions[:, [left, right]]
    velocity = np.zeros((len(feet), 2), dtype=np.float64)
    if len(feet) > 1:
        velocity[1:] = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * fps
        velocity[0] = velocity[1]
    ground = float(np.percentile(feet[..., 2], 1.0))
    contacts = (velocity < 0.01) & (feet[..., 2] < ground + 0.05)
    return np.stack([_remove_short_true_runs(contacts[:, i]) for i in range(2)], axis=1)


def _inspect_window(
    path: str | Path,
    motion,
    world_rotations: np.ndarray,
    positions: np.ndarray,
    frame_start: int,
    frame_end: int,
) -> SourceFeatures:
    world_rotations = world_rotations[frame_start:frame_end]
    positions = positions[frame_start:frame_end]
    dt = motion.frame_time
    duration = len(positions) * dt
    root_step = np.linalg.norm(np.diff(positions[:, 0, :2], axis=0), axis=-1)
    root_path = float(root_step.sum())
    root_speed = float(root_step.mean() / dt) if len(root_step) else 0.0
    root_yaw = np.unwrap(yaw_from_matrix(quaternion_wxyz_to_matrix(world_rotations[:, 0])))
    yaw_range = float(root_yaw.max() - root_yaw.min())
    try:
        left_wrist = _find_joint(motion.joint_names, ("LeftHand", "LeftWrist", "L_Wrist"))
        right_wrist = _find_joint(motion.joint_names, ("RightHand", "RightWrist", "R_Wrist"))
        wrist_range = float(
            np.mean(
                [
                    np.ptp(positions[:, left_wrist], axis=0).max(),
                    np.ptp(positions[:, right_wrist], axis=0).max(),
                ]
            )
        )
    except ValueError:
        wrist_range = 0.0
    try:
        contacts = foot_contacts(positions, motion.joint_names, motion.fps)
        transitions = int(np.count_nonzero(np.diff(contacts.astype(np.int8), axis=0)))
    except ValueError:
        transitions = 0
    reasons: list[str] = []
    if not 10.0 <= duration <= 30.0:
        reasons.append("duration")
    if not (root_path >= 1.0 or yaw_range >= 0.35):
        reasons.append("locomotion")
    if wrist_range < 0.15:
        reasons.append("arm_motion")
    if transitions < 2:
        reasons.append("stance_swing")
    return SourceFeatures(
        sequence_id=f"{Path(path).stem}_f{frame_start:06d}_{frame_end:06d}",
        source_file=str(Path(path)),
        frame_start=frame_start,
        frame_end=frame_end,
        frames=frame_end - frame_start,
        fps=motion.fps,
        duration_s=duration,
        root_path_m=root_path,
        root_mean_speed_mps=root_speed,
        root_yaw_range_rad=yaw_range,
        wrist_range_m=wrist_range,
        contact_transitions=transitions,
        eligible=not reasons,
        rejection_reason=";".join(reasons),
    )


def inspect_source(
    path: str | Path,
    position_scale: float = 0.01,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> SourceFeatures:
    motion = parse_bvh(path)
    _, world_rotations, positions, _ = forward_kinematics(
        motion, position_scale, coordinate_transform=LAFAN_TO_Z_UP
    )
    end = motion.frame_count if frame_end is None else frame_end
    if not 0 <= frame_start < end <= motion.frame_count:
        raise ValueError("Invalid source frame range")
    return _inspect_window(path, motion, world_rotations, positions, frame_start, end)


def select_pilot(
    lafan_root: str | Path, selection_csv: str | Path, position_scale: float = 0.01
) -> SourceFeatures:
    files = sorted(Path(lafan_root).rglob("*.bvh"), key=lambda path: path.as_posix().lower())
    if not files:
        raise FileNotFoundError(f"No BVH files found below {lafan_root}")
    features: list[SourceFeatures] = []
    semantic_exclusions = ("aiming", "fall", "ground", "obstacles", "push")
    for path in files:
        motion = parse_bvh(path)
        _, world_rotations, positions, _ = forward_kinematics(
            motion, position_scale, coordinate_transform=LAFAN_TO_Z_UP
        )
        window = max(1, int(round(20.0 * motion.fps)))
        stride = max(1, int(round(10.0 * motion.fps)))
        if motion.frame_count < window:
            continue
        starts = list(range(0, motion.frame_count - window + 1, stride))
        final_start = motion.frame_count - window
        if final_start not in starts:
            starts.append(final_start)
        for start in starts:
            value = _inspect_window(
                path, motion, world_rotations, positions, start, start + window
            )
            if Path(path).stem.lower().startswith(semantic_exclusions):
                value = SourceFeatures(
                    **{
                        **asdict(value),
                        "eligible": False,
                        "rejection_reason": ";".join(
                            filter(None, (value.rejection_reason, "semantic_exclusion"))
                        ),
                    }
                )
            features.append(value)
    eligible = [value for value in features if value.eligible]
    if not eligible:
        raise RuntimeError("No LAFAN sequence satisfies the frozen source-only criteria")
    selected = min(
        eligible,
        key=lambda value: (
            abs(value.duration_s - 20.0),
            Path(value.source_file).name.lower(),
            value.frame_start,
        ),
    )
    output = Path(selection_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(features[0])))
        writer.writeheader()
        writer.writerows(asdict(value) for value in features)
    return selected


def crop_bvh(
    source_path: str | Path, output_path: str | Path, frame_start: int, frame_end: int
) -> None:
    """Write an exact frame-range BVH derivative without changing channel values."""
    lines = Path(source_path).read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        motion_index = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as error:
        raise ValueError("BVH has no MOTION section") from error
    if motion_index + 3 > len(lines):
        raise ValueError("BVH MOTION header is truncated")
    total_frames = int(lines[motion_index + 1].split(":", 1)[1].strip())
    if not 0 <= frame_start < frame_end <= total_frames:
        raise ValueError("Invalid BVH crop range")
    data_start = motion_index + 3
    frame_lines = [line for line in lines[data_start:] if line.strip()]
    if len(frame_lines) != total_frames:
        raise ValueError("BVH frame lines do not match declared frame count")
    result = (
        lines[: motion_index + 1]
        + [f"Frames: {frame_end - frame_start}", lines[motion_index + 2]]
        + frame_lines[frame_start:frame_end]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def canonicalize_source(
    source_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    position_scale: float = 0.01,
    origin_path: str | Path | None = None,
    origin_frame_start: int = 0,
    origin_frame_end: int | None = None,
    repo_root: str | Path | None = None,
) -> CanonicalHuman:
    motion = parse_bvh(source_path)
    local, world, positions, root = forward_kinematics(
        motion, position_scale, coordinate_transform=LAFAN_TO_Z_UP
    )
    contacts = foot_contacts(positions, motion.joint_names, motion.fps)
    digest = sha256_file(source_path)
    canonical = CanonicalHuman(
        joint_names=np.asarray(motion.joint_names),
        parent_indices=motion.parent_indices,
        local_rotations=local,
        world_rotations=world,
        world_positions=positions,
        root_translation=root,
        fps=motion.fps,
        timestamps=np.arange(motion.frame_count, dtype=np.float64) / motion.fps,
        foot_contact_labels=contacts,
        source_sha256=digest,
    )
    canonical.save(output_path)
    origin = Path(origin_path or source_path)
    root = Path(repo_root).resolve() if repo_root is not None else None

    def portable(path: Path) -> str:
        if root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)

    atomic_write_yaml(
        manifest_path,
        {
            "sequence_id": Path(source_path).stem,
            "source_file": portable(origin),
            "source_sha256": sha256_file(origin),
            "frame_start": int(origin_frame_start),
            "frame_end": int(origin_frame_end or (origin_frame_start + motion.frame_count)),
            "cropped_source_file": portable(Path(source_path)),
            "cropped_sha256": digest,
            "fps": float(motion.fps),
            "num_frames": int(motion.frame_count),
            "duration_s": float(motion.frame_count / motion.fps),
            "selection_rule": (
                "source-only filters; minimize abs(duration-20s); lexicographic filename tie-break"
            ),
            "selected_before_any_retarget_run": True,
            "canonical_path": portable(Path(output_path)),
        },
    )
    return canonical
