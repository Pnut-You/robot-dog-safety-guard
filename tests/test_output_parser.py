import pytest

from app.inference import parse_prediction


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACCEPT", "ACCEPT"),
        ("REJECT", "REJECT"),
        ("  ACCEPT  ", "ACCEPT"),
        ("\nREJECT\n", "REJECT"),
        ("accept", "INVALID"),
        ("Reject", "INVALID"),
        ("ACCEPT，因为这是支持的动作", "INVALID"),
        ("", "INVALID"),
        ("   \n", "INVALID"),
        ("YES", "INVALID"),
    ],
)
def test_parse_prediction_is_strict(raw, expected):
    assert parse_prediction(raw) == expected
