import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_DATA_DIR = PROJECT_ROOT / "docs" / "results"
LEGACY_DATA_DIR = PROJECT_ROOT.parent / "prescription_data"


def find_latest_json(directory: Path) -> Path | None:
    if not directory.exists():
        return None

    json_files = list(directory.glob("*.json"))
    if not json_files:
        return None

    return max(json_files, key=lambda path: path.stat().st_mtime)


def safe_get(dictionary, key, default=None):
    if not isinstance(dictionary, dict):
        return default
    return dictionary.get(key, default)


def print_header(title: str) -> None:
    print()
    print(f"========== {title} ==========")


def round_value(value):
    if isinstance(value, (int, float)):
        return round(value, 2)
    return value


def quality_text(rom_value):
    if rom_value is None:
        return "未能计算 ROM，建议检查关键点可见度后重新录制。"
    if rom_value < 15:
        return "ROM 偏小，建议重新录制。"
    if rom_value < 30:
        return "ROM 勉强可用，建议再录一版更稳定的结果。"
    return "ROM 较明显，这份结果可以作为后续模板参考。"


def main() -> None:
    print("正在读取最新处方结果...")

    latest_file = find_latest_json(PRIMARY_DATA_DIR)
    source_dir = PRIMARY_DATA_DIR

    if latest_file is None:
        print(f"本地结果目录中没有找到 JSON：{PRIMARY_DATA_DIR}")
        latest_file = find_latest_json(LEGACY_DATA_DIR)
        source_dir = LEGACY_DATA_DIR
        if latest_file is not None:
            print(f"已回退到旧目录读取：{LEGACY_DATA_DIR}")

    if latest_file is None:
        print("没有找到任何 JSON 文件。")
        print("请先运行 `local_result_sink.py` 和 `record_prescription_http.py` 完成一次保存。")
        return

    print(f"读取文件：{latest_file}")
    print(f"来源目录：{source_dir}")

    with latest_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    patient_id = data.get("patient_id")
    action_name = data.get("action_name")
    created_at = data.get("created_at")

    clinical = data.get("clinical_baseline", {})
    runtime_meta = data.get("runtime_meta", {})

    frame_count = safe_get(clinical, "frame_count")
    duration_seconds = safe_get(clinical, "duration_seconds")
    min_flexion = safe_get(clinical, "min_knee_flexion_angle")
    max_flexion = safe_get(clinical, "max_knee_flexion_angle")
    rom_flexion = safe_get(clinical, "rom_flexion")

    old_min_angle = safe_get(clinical, "min_left_knee_angle")
    old_max_angle = safe_get(clinical, "max_left_knee_angle")
    old_rom = safe_get(clinical, "rom")

    invalid_frame_count = safe_get(runtime_meta, "invalid_frame_count", 0)
    side_mode = safe_get(runtime_meta, "side_mode", "unknown")
    prefer_3d = safe_get(runtime_meta, "prefer_3d_world_angle", "unknown")
    model_complexity = safe_get(runtime_meta, "model_complexity", "unknown")
    result_format = safe_get(runtime_meta, "result_format", "legacy_full")

    print_header("基本信息")
    print("患者编号：", patient_id)
    print("动作名称：", action_name)
    print("创建时间：", created_at)

    print_header("录制质量")
    print("有效帧数：", frame_count)
    print("动作时长：", round_value(duration_seconds), "秒")
    print("无效帧数：", invalid_frame_count)
    print("侧别模式：", side_mode)
    print("优先使用 3D：", prefer_3d)
    print("模型复杂度：", model_complexity)
    print("结果格式：", result_format)

    print_header("膝关节结果")
    if rom_flexion is not None:
        print("最小屈曲角：", round_value(min_flexion), "度")
        print("最大屈曲角：", round_value(max_flexion), "度")
        print("ROM：", round_value(rom_flexion), "度")
        print("结果判断：", quality_text(rom_flexion))
    else:
        print("没有找到新版屈曲角字段，尝试读取旧版字段。")
        print("最小原始膝角：", round_value(old_min_angle))
        print("最大原始膝角：", round_value(old_max_angle))
        print("ROM：", round_value(old_rom))
        print("结果判断：", quality_text(old_rom))

    print()
    print("读取完成。")


if __name__ == "__main__":
    main()
