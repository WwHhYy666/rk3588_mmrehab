from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rehab_app.services.result_storage import save_prescription_artifacts

from rehab_app.services.result_storage import save_prescription_artifacts


def _payload(record_role: str) -> dict[str, object]:
    return {
        "patient_id": "patient_001",
        "record_role": record_role,
        "action_name": "seated_knee_extension",
        "runtime_meta": {
            "record_role": record_role,
            "side_mode": "auto",
            "result_format": "compact_v1",
            "invalid_frame_count": 0,
        },
        "clinical_baseline": {
            "frame_count": 3,
            "duration_seconds": 1.2,
            "min_knee_flexion_angle": 10.0,
            "max_knee_flexion_angle": 50.0,
            "rom_flexion": 40.0,
        },
        "template_frames": [],
    }


def test_save_artifacts_separates_doctor_and_patient_roles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docs_dir = Path(tmp)
        doctor = save_prescription_artifacts(_payload("doctor_template"), docs_dir=docs_dir)
        patient = save_prescription_artifacts(_payload("patient_attempt"), docs_dir=docs_dir)

        doctor_path = Path(str(doctor["saved_path"]))
        patient_path = Path(str(patient["saved_path"]))

        assert doctor_path.parent.name == "doctor_templates"
        assert patient_path.parent.name == "patient_attempts"
        assert doctor_path.name.startswith("doctor_template_patient_001_")
        assert patient_path.name.startswith("patient_attempt_patient_001_")
        assert json.loads(doctor_path.read_text(encoding="utf-8"))["record_role"] == "doctor_template"
        assert json.loads(patient_path.read_text(encoding="utf-8"))["record_role"] == "patient_attempt"
        assert doctor["summary"]["record_role"] == "doctor_template"
        assert patient["summary"]["record_role"] == "patient_attempt"


if __name__ == "__main__":
    test_save_artifacts_separates_doctor_and_patient_roles()
    print("result storage tests passed")
