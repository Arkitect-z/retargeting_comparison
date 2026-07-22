import numpy as np

from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE
from retargeting_comparison.reporting import REPORTS
from retargeting_comparison.schemas import CanonicalG1
from retargeting_comparison.validation import _core_motion_check


def test_delivery_contract_lists_all_required_markdown() -> None:
    assert len(REPORTS) == 8
    assert "PRESENTATION.md" in REPORTS
    assert FULL_LAFAN_STOP_MESSAGE.endswith("USER APPROVAL.")


def test_core_motion_check_returns_json_native_bool() -> None:
    qpos = np.zeros((2, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    motion = CanonicalG1(
        qpos=qpos,
        fps=30.0,
        source_frame_idx=np.arange(2),
        valid=np.ones(2, dtype=bool),
        per_frame_solve_time_s=np.zeros(2),
        metadata={"completion_status": "succeeded"},
    )

    result = _core_motion_check(motion, source_frame_count=2)

    assert result is True
    assert type(result) is bool
