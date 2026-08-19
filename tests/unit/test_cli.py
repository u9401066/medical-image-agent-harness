from __future__ import annotations

from medical_image_harness.cli import _validate


def test_cli_rejects_nonstandard_nan_json(tmp_path, capsys) -> None:
    path = tmp_path / "nan-result.json"
    path.write_text('{"x": NaN}', encoding="utf-8")

    status = _validate(path)

    assert status == 2
    assert "non-finite JSON number is forbidden" in capsys.readouterr().out
