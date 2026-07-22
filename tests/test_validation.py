from retargeting_comparison.constants import FULL_LAFAN_STOP_MESSAGE
from retargeting_comparison.reporting import REPORTS


def test_delivery_contract_lists_all_required_markdown() -> None:
    assert len(REPORTS) == 8
    assert "PRESENTATION.md" in REPORTS
    assert FULL_LAFAN_STOP_MESSAGE.endswith("USER APPROVAL.")
