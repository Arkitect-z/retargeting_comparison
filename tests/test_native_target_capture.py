from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from retargeting_comparison.io_utils import sha256_file
from retargeting_comparison.native_target_capture import (
    CAPTURE_CLASS,
    HOLOSOMA_LAFAN_ORDER,
    PROTOMOTIONS_V3_LABELS,
    aggregate_captures,
    gmr_scaled_positions,
    holosoma_lafan_preprocess,
    protomotions_v2_positions,
    protomotions_v3_positions,
    tensor_sha256,
    write_deterministic_npz,
)
from retargeting_comparison.schemas import CanonicalHuman


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_ID = "dance1_subject1_f000000_000600"
CAPTURE_ROOT = ROOT / "runs" / SEQUENCE_ID / "native-pre-solver-targets"
EXPECTED_TENSOR_SHA256 = {
    "gmr": "f5be367363cd1a89cb7c4099e9606d117551e01e937ad7d04d6771ffb780d4bc",
    "omniretarget": "0d56c519cd130c16772b189f57cad7b0126b2ffe58535e656789c520bba90f82",
    "protomotions-v2.3": "84949253060eb4d3bb79ab29565fd1a436b2af2d0aeadf078b06fecc71e20575",
    "protomotions-v3": "4a5cb617bb04f4ff8772d986baafa8ae1c06e8036de315996f71d039565475af",
}


def test_tensor_hash_includes_dtype_shape_and_values() -> None:
    value = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    assert tensor_sha256(value) == tensor_sha256(value.copy())
    assert tensor_sha256(value) != tensor_sha256(value.astype(np.float32))
    changed = value.copy()
    changed[0, 0, 0] += 1.0
    assert tensor_sha256(value) != tensor_sha256(changed)


def test_deterministic_npz_is_byte_reproducible_and_pickle_free(tmp_path: Path) -> None:
    arrays = {
        "positions": np.arange(18, dtype=np.float64).reshape(2, 3, 3),
        "labels": np.asarray(["root", "left", "right"]),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert sha256_file(first) == sha256_file(second)
    with np.load(first, allow_pickle=False) as archive:
        assert np.array_equal(archive["positions"], arrays["positions"])
        assert np.array_equal(archive["labels"], arrays["labels"])


def test_frozen_scale_and_ground_formulas() -> None:
    root = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    body = root + np.asarray([[0.5, -0.5, 1.0], [1.0, -1.0, 2.0]])
    actual = gmr_scaled_positions(body, root, np.asarray(0.75), np.asarray(0.9))
    assert np.allclose(actual, root * 0.9 + (body - root) * 0.75)

    holo = np.zeros((2, len(HOLOSOMA_LAFAN_ORDER), 3), dtype=np.float64)
    holo[:, :, 2] = 1.0
    holo[:, HOLOSOMA_LAFAN_ORDER.index("LeftToeBase"), 2] = 0.2
    holo[:, HOLOSOMA_LAFAN_ORDER.index("RightToeBase"), 2] = 0.3
    processed, ground = holosoma_lafan_preprocess(holo)
    assert ground == pytest.approx(0.1)
    assert processed[0, HOLOSOMA_LAFAN_ORDER.index("LeftToeBase"), 2] == pytest.approx(
        0.1 * (1.27 / 1.7)
    )
    assert processed[0, HOLOSOMA_LAFAN_ORDER.index("Spine1"), 2] == pytest.approx(
        0.84 * (1.27 / 1.7)
    )

    v2 = np.asarray([[[1.0, 2.0, 3.0]]])
    assert np.allclose(protomotions_v2_positions(v2), [[[0.75, 2.0, 2.4]]])

    v3 = np.zeros((2, 18, 3), dtype=np.float64)
    v3[:, 0] = [1.0, 2.0, 3.0]
    v3[:, 1:9] = v3[:, :1] + 1.0
    v3[:, 9:] = v3[:, :1] + 2.0
    scaled = protomotions_v3_positions(v3)
    assert np.allclose(scaled[:, 0], [[0.9, 1.8, 2.55]] * 2)
    assert np.allclose(scaled[:, 1] - scaled[:, 0], [[0.9, 0.9, 0.85]] * 2)
    assert np.allclose(scaled[:, 9] - scaled[:, 0], [[1.8, 1.8, 1.6]] * 2)


def _captures_available() -> bool:
    return all(
        (CAPTURE_ROOT / method / "capture.json").is_file()
        and (CAPTURE_ROOT / method / "targets.npz").is_file()
        for method in ("gmr", "omniretarget", "protomotions-v2.3", "protomotions-v3")
    )


@pytest.mark.skipif(not _captures_available(), reason="ignored native captures are absent")
def test_all_pilot_captures_are_exact_hashed_and_solver_free() -> None:
    expected = {"gmr", "omniretarget", "protomotions-v2.3", "protomotions-v3"}
    manifest = list(
        csv.DictReader((ROOT / "manifests/native_pre_solver_targets.csv").open())
    )
    assert {row["method"] for row in manifest} == expected
    for row in manifest:
        assert row["capture_class"] == CAPTURE_CLASS
        assert row["exact_solver_input"] == "true"
        assert row["normalized_projection"] == "false"
        assert row["capture_generation_solver_invoked"] == "false"
        assert row["boundary_observed_during_formal_run"] == "true"
        assert row["reconstruction_matches_runtime"] == "true"
        assert row["runtime_witness_solver_invoked"] == "true"
        assert row["acquisition_mode"] in {
            "native_runtime_target_setter_replay_verified",
            "runtime_hash_verified_reconstruction",
            "official_loader_runtime_hash_verified",
        }
        expected_geometry_class = (
            "exact_pre_solver_position_source"
            if row["method"] == "omniretarget"
            else "exact_solver_input"
        )
        assert row["geometry_class"] == expected_geometry_class
        assert int(row["frames"]) == 600
        artifact = ROOT / row["target_artifact"]
        assert sha256_file(artifact) == row["target_artifact_sha256"]
        with np.load(artifact, allow_pickle=False) as archive:
            tensor = np.asarray(archive[row["tensor_key"]])
            labels = np.asarray(archive[row["tensor_label_key"]]).astype(str)
        assert len(labels) == tensor.shape[1]
        assert tensor_sha256(tensor) == row["tensor_sha256"]
        assert tensor_sha256(tensor) == row["runtime_tensor_sha256"]
        assert row["tensor_sha256"] == EXPECTED_TENSOR_SHA256[row["method"]]
        for path_field, hash_field in (
            ("source_path", "source_sha256"),
            ("config_path", "config_sha256"),
            ("robot_asset_path", "robot_asset_sha256"),
            ("implementation_path", "implementation_sha256"),
        ):
            assert sha256_file(ROOT / row[path_field]) == row[hash_field]


@pytest.mark.skipif(not _captures_available(), reason="ignored native captures are absent")
def test_captured_tensors_reproduce_each_frozen_formula() -> None:
    human = CanonicalHuman.load(
        ROOT / "source/canonical_human" / f"{SEQUENCE_ID}.npz"
    )
    human_indices = {
        name: index for index, name in enumerate(human.joint_names.astype(str))
    }

    gmr_meta = json.loads((CAPTURE_ROOT / "gmr/capture.json").read_text())
    with np.load(CAPTURE_ROOT / "gmr/targets.npz", allow_pickle=False) as archive:
        gmr = np.asarray(archive["position_targets"])
        labels = np.asarray(archive["target_labels"]).astype(str)
    scales = gmr_meta["human_scale_table"]
    expected_gmr = np.empty_like(gmr)
    root = human.world_positions[:, human_indices["Hips"]]
    for index, label in enumerate(labels):
        human_name = label.split("<-")[1]
        source_name = "LeftFoot" if human_name == "LeftFootMod" else human_name
        source_name = "RightFoot" if human_name == "RightFootMod" else source_name
        body = human.world_positions[:, human_indices[source_name]]
        expected_gmr[:, index] = gmr_scaled_positions(
            body,
            root,
            np.asarray(scales[human_name]),
            np.asarray(scales["Hips"]),
        )
    assert np.allclose(gmr, expected_gmr, atol=1e-12, rtol=0.0)
    assert np.array_equal(gmr[:, :14], gmr[:, 14:])

    with np.load(CAPTURE_ROOT / "omniretarget/targets.npz", allow_pickle=False) as archive:
        holo_all = np.asarray(archive["human_joint_positions"])
        holo_mapped = np.asarray(archive["mapped_position_targets"])
        holo_labels = np.asarray(archive["human_joint_labels"]).astype(str)
        mapped_labels = np.asarray(archive["mapped_position_labels"]).astype(str)
        laplacian = np.asarray(archive["solver_target_laplacian"])
    source_holo = np.load(
        ROOT
        / "source_adapters/omniretarget"
        / SEQUENCE_ID
        / "pilot_canonical.npy"
    )[..., [0, 2, 1]]
    expected_holo, _ = holosoma_lafan_preprocess(source_holo)
    assert tuple(holo_labels) == HOLOSOMA_LAFAN_ORDER
    assert np.array_equal(holo_all, expected_holo)
    mapped_indices = [HOLOSOMA_LAFAN_ORDER.index(name) for name in mapped_labels]
    assert np.array_equal(holo_mapped, holo_all[:, mapped_indices])
    assert laplacian.shape == (600, 240, 3)

    with np.load(CAPTURE_ROOT / "protomotions-v2.3/targets.npz", allow_pickle=False) as archive:
        v2 = np.asarray(archive["position_targets"])
        source_labels = np.asarray(archive["human_joint_labels"]).astype(str)
    v2_source = human.world_positions[
        :, [human_indices[name] for name in source_labels]
    ]
    assert np.array_equal(v2, protomotions_v2_positions(v2_source))

    raw_v3 = np.load(
        ROOT / "source_adapters/protomotions_v3" / SEQUENCE_ID / "keypoints.npy",
        allow_pickle=True,
    ).item()
    with np.load(CAPTURE_ROOT / "protomotions-v3/targets.npz", allow_pickle=False) as archive:
        v3 = np.asarray(archive["target_keypoints"])
        v3_labels = np.asarray(archive["target_labels"]).astype(str)
    assert tuple(v3_labels) == PROTOMOTIONS_V3_LABELS
    assert np.array_equal(v3, protomotions_v3_positions(raw_v3["positions"]))


@pytest.mark.skipif(not _captures_available(), reason="ignored native captures are absent")
def test_aggregate_tables_are_byte_reproducible() -> None:
    first = aggregate_captures(ROOT)
    second = aggregate_captures(ROOT)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["geometry_sha256"] == second["geometry_sha256"]
    assert first["geometry_rows"] == 75
    assert first["all_exact_solver_input"] is True
    assert first["all_formal_boundaries_observed"] is True
    assert first["all_reconstructions_match_runtime"] is True
    assert first["normalized_projection_rows"] == 0
    assert first["capture_generation_solver_invocations"] == 0
    assert first["runtime_witness_solver_invocations"] == 4
