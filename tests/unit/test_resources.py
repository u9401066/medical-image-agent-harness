from medical_image_harness.resources import (
    load_modality_prompt,
    load_skill,
    skill_sha256,
)


def test_skill_and_modality_prompt_load_from_single_source() -> None:
    skill = load_skill()
    prompt = load_modality_prompt("dicom-ekg-analysis")
    assert "name: medical-image-reading" in skill
    assert "Blind systematic pass" in prompt
    assert "Lead-conditioned blind pass" in prompt
    assert len(skill_sha256()) == 64
