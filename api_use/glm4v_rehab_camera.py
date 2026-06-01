from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4v-flash"

DEFAULT_SYSTEM_PROMPT = """你是一个部署在智能居家医疗嵌入式系统上的多模态康复体操辅助模型。
系统会持续接入本地摄像头画面，并把每一个体操动作循环中最关键的一帧传给你。你需要结合图像和文本输入，对康复体操动作是否标准做分析。

你的任务：
1. 判断当前动作是否接近标准姿态，给出 0-100 的标准度评分。
2. 针对关键指标做视觉估计。例如“缓慢高抬腿”需要估计大腿与水平线之间的夹角、膝关节是否过度弯曲、躯干是否后仰、支撑腿是否稳定。
3. 一组动作结束后，根据动作类型、次数、持续时间、体重等信息，估计运动消耗的卡路里，并说明估计依据。
4. 支持用户的语音补充或提问，回答要短、清楚、适合终端显示和语音播报。

注意：
- 你不是医生，不能做诊断。只给居家康复训练的动作反馈、安全提醒和复查建议。
- 如果画面遮挡、角度不足或无法看清身体关键部位，要明确说明置信度低，并建议调整摄像头或站位。
- 输出优先使用中文，格式紧凑。"""

KEYFRAME_USER_PROMPT = """当前正在进行居家康复体操动作分析。

动作名称：{exercise_name}
当前动作循环编号：第 {rep_index} 次
目标次数：{target_reps}
已训练时长：{elapsed_seconds:.1f} 秒
用户体重：{weight_kg:.1f} kg
关键帧提取信息：motion={motion_score:.2f}, displacement={displacement_score:.2f}, reason={reason}
用户语音补充：{voice_text}

请只分析这一帧对应的动作关键姿态，重点输出：
1. 标准度评分
2. 关键指标估计，包含角度或相对位置；如果是高抬腿，请估计大腿与水平线夹角
3. 主要问题和风险
4. 下一次动作的简短纠正建议"""

SUMMARY_USER_PROMPT = """一组居家康复体操已经结束，请根据历史关键帧分析做总结。

动作名称：{exercise_name}
完成次数：{rep_count}
目标次数：{target_reps}
总时长：{elapsed_seconds:.1f} 秒
用户体重：{weight_kg:.1f} kg
用户语音补充：{voice_text}

历史关键帧分析：
{history}

请输出：
1. 本组动作整体标准度和主要趋势
2. 估计消耗卡路里 kcal，并说明估计依据和不确定性
3. 下一组训练建议
4. 是否存在需要停止训练或咨询医生的风险信号"""

VOICE_QUESTION_PROMPT = """用户通过语音提出了一个问题或补充了训练信息。

动作名称：{exercise_name}
已完成次数：{rep_count}
已训练时长：{elapsed_seconds:.1f} 秒
用户语音文本：{voice_text}

如果附带了当前摄像头画面，请结合画面回答；如果画面不足，请说明需要用户调整站位或摄像头。回答要简短。"""


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "缺少 OpenCV。请先安装：pip install -r api_use/requirements-glm4v.txt"
        )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def load_voice_components():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from voice_qwen_core import (
        Recorder,
        SpeechRecognitionAsr,
        audio_to_wav_bytes,
        normalize_device,
    )

    return Recorder, SpeechRecognitionAsr, audio_to_wav_bytes, normalize_device


def short_text(text: str, max_chars: int = 1200) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def contains_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


@dataclass
class KeyFrame:
    frame: np.ndarray
    motion_score: float
    displacement_score: float
    duration_seconds: float
    reason: str


class MotionKeyFrameExtractor:
    """Small state machine for laptop prototyping.

    It detects one action cycle as a burst of motion followed by a short stable
    period. The emitted frame is the frame farthest from the pre-motion
    baseline, which is usually closer to the action apex than the fastest frame.
    """

    def __init__(
        self,
        motion_threshold: float = 6.0,
        stable_ratio: float = 0.55,
        stable_frames: int = 5,
        cooldown_frames: int = 6,
        min_cycle_seconds: float = 0.7,
        max_cycle_seconds: float = 6.0,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.stable_threshold = motion_threshold * stable_ratio
        self.stable_frames = stable_frames
        self.cooldown_frames = cooldown_frames
        self.min_cycle_seconds = min_cycle_seconds
        self.max_cycle_seconds = max_cycle_seconds

        self.prev_gray: np.ndarray | None = None
        self.baseline_gray: np.ndarray | None = None
        self.in_motion = False
        self.motion_start = 0.0
        self.stable_count = 0
        self.cooldown_count = 0
        self.best_frame: np.ndarray | None = None
        self.best_motion = 0.0
        self.best_displacement = 0.0

    def update(self, frame: np.ndarray, now: float) -> tuple[KeyFrame | None, float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.baseline_gray = gray
            return None, 0.0, 0.0

        motion = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
        self.prev_gray = gray

        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            if motion <= self.stable_threshold:
                self.baseline_gray = gray
            return None, motion, 0.0

        if not self.in_motion:
            if motion >= self.motion_threshold:
                self.in_motion = True
                self.motion_start = now
                self.stable_count = 0
                self.best_frame = frame.copy()
                self.best_motion = motion
                self.best_displacement = 0.0
            else:
                self.baseline_gray = gray
            return None, motion, 0.0

        displacement = motion
        if self.baseline_gray is not None:
            displacement = float(np.mean(cv2.absdiff(gray, self.baseline_gray)))

        if displacement >= self.best_displacement:
            self.best_frame = frame.copy()
            self.best_motion = motion
            self.best_displacement = displacement

        if motion <= self.stable_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        elapsed = now - self.motion_start
        stable_done = elapsed >= self.min_cycle_seconds and self.stable_count >= self.stable_frames
        timeout_done = elapsed >= self.max_cycle_seconds
        if stable_done or timeout_done:
            reason = "stable_after_motion" if stable_done else "max_cycle_timeout"
            emitted = KeyFrame(
                frame=self.best_frame if self.best_frame is not None else frame.copy(),
                motion_score=self.best_motion,
                displacement_score=self.best_displacement,
                duration_seconds=elapsed,
                reason=reason,
            )
            self._reset_after_emit(gray)
            return emitted, motion, displacement

        return None, motion, displacement

    def force_emit(self, frame: np.ndarray, reason: str = "manual") -> KeyFrame:
        emitted = KeyFrame(
            frame=frame.copy(),
            motion_score=self.best_motion,
            displacement_score=self.best_displacement,
            duration_seconds=max(0.0, time.monotonic() - self.motion_start) if self.in_motion else 0.0,
            reason=reason,
        )
        return emitted

    def _reset_after_emit(self, gray: np.ndarray) -> None:
        self.in_motion = False
        self.stable_count = 0
        self.cooldown_count = self.cooldown_frames
        self.baseline_gray = gray
        self.best_frame = None
        self.best_motion = 0.0
        self.best_displacement = 0.0


class Glm4vClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._post_json(payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"GLM API response missing choices/message/content: {response}") from exc
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"GLM API HTTP {exc.code}: {detail}")
                if exc.code < 500 or attempt >= self.max_retries:
                    raise last_error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
            time.sleep(min(2.0 * (attempt + 1), 6.0))

        raise RuntimeError(f"GLM API request failed after retries: {last_error}")


class DryRunClient:
    def chat(self, messages: list[dict[str, Any]]) -> str:
        text_chars = 0
        image_count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                text_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_chars += len(item.get("text", ""))
                    if item.get("type") == "image_url":
                        image_count += 1
        return f"[dry-run] 已构建 GLM 请求：文本约 {text_chars} 字，图片 {image_count} 张。"


class RehabAssistant:
    def __init__(
        self,
        client: Glm4vClient | DryRunClient,
        exercise_name: str,
        target_reps: int,
        weight_kg: float,
        image_mode: str,
        jpeg_quality: int,
        image_max_width: int,
        system_prompt: str,
    ) -> None:
        self.client = client
        self.exercise_name = exercise_name
        self.target_reps = target_reps
        self.weight_kg = weight_kg
        self.image_mode = image_mode
        self.jpeg_quality = jpeg_quality
        self.image_max_width = image_max_width
        self.system_prompt = system_prompt
        self.started_at = time.monotonic()
        self.rep_count = 0
        self.history: list[str] = []
        self.latest_frame: np.ndarray | None = None
        self.summary_done = False

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def analyze_keyframe(self, keyframe: KeyFrame, voice_text: str = "") -> str:
        self.rep_count += 1
        self.latest_frame = keyframe.frame.copy()
        prompt = KEYFRAME_USER_PROMPT.format(
            exercise_name=self.exercise_name,
            rep_index=self.rep_count,
            target_reps=self.target_reps,
            elapsed_seconds=self.elapsed_seconds,
            weight_kg=self.weight_kg,
            motion_score=keyframe.motion_score,
            displacement_score=keyframe.displacement_score,
            reason=keyframe.reason,
            voice_text=voice_text or "无",
        )
        answer = self._chat_with_frame(keyframe.frame, prompt)
        self.history.append(f"第 {self.rep_count} 次：{short_text(answer, 500)}")
        print("\n" + "=" * 72)
        print(f"GLM-4V-Flash 第 {self.rep_count} 次关键帧分析")
        print("=" * 72)
        print(answer)
        print("=" * 72 + "\n")
        return answer

    def answer_voice_question(self, voice_text: str, frame: np.ndarray | None = None) -> str:
        if frame is not None:
            self.latest_frame = frame.copy()
        prompt = VOICE_QUESTION_PROMPT.format(
            exercise_name=self.exercise_name,
            rep_count=self.rep_count,
            elapsed_seconds=self.elapsed_seconds,
            voice_text=voice_text,
        )
        if self.latest_frame is not None:
            answer = self._chat_with_frame(self.latest_frame, prompt)
        else:
            answer = self._chat_text_only(prompt)
        print("\n" + "-" * 72)
        print("GLM-4V-Flash 语音交互回复")
        print("-" * 72)
        print(answer)
        print("-" * 72 + "\n")
        return answer

    def summarize_set(self, voice_text: str = "") -> str:
        history = "\n".join(self.history[-12:]) if self.history else "暂无关键帧分析。"
        prompt = SUMMARY_USER_PROMPT.format(
            exercise_name=self.exercise_name,
            rep_count=self.rep_count,
            target_reps=self.target_reps,
            elapsed_seconds=self.elapsed_seconds,
            weight_kg=self.weight_kg,
            voice_text=voice_text or "无",
            history=history,
        )
        answer = self._chat_text_only(prompt)
        self.summary_done = True
        print("\n" + "#" * 72)
        print("GLM-4V-Flash 本组动作总结")
        print("#" * 72)
        print(answer)
        print("#" * 72 + "\n")
        return answer

    def _chat_text_only(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.client.chat(messages)

    def _chat_with_frame(self, frame: np.ndarray, prompt: str) -> str:
        image_payload = encode_frame_to_image_url(
            frame,
            max_width=self.image_max_width,
            jpeg_quality=self.jpeg_quality,
            image_mode=self.image_mode,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_payload}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self.client.chat(messages)


class VoiceCommandWorker(threading.Thread):
    def __init__(
        self,
        output_queue: queue.Queue[str],
        duration: float,
        interval: float,
        sample_rate: int,
        input_device: str | None,
        asr_backend: str,
        language: str,
        whisper_model: str,
    ) -> None:
        super().__init__(daemon=True)
        self.output_queue = output_queue
        self.duration = duration
        self.interval = interval
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.asr_backend = asr_backend
        self.language = language
        self.whisper_model = whisper_model
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            Recorder, SpeechRecognitionAsr, audio_to_wav_bytes, normalize_device = load_voice_components()
            recorder = Recorder(
                sample_rate=self.sample_rate,
                input_device=normalize_device(self.input_device),
            )
            asr = SpeechRecognitionAsr(
                backend=self.asr_backend,
                language=self.language,
                whisper_model=self.whisper_model,
            )
            print("[voice] 语音命令监听已启动。")
            while not self.stop_event.is_set():
                audio = recorder.record_array(self.duration)
                wav_data = audio_to_wav_bytes(audio, self.sample_rate)
                transcript = asr.transcribe_wav_bytes(wav_data).strip()
                if transcript:
                    print(f"[voice] {transcript}")
                    self.output_queue.put(transcript)
                wait_seconds = max(0.1, self.interval - self.duration)
                self.stop_event.wait(wait_seconds)
        except Exception as exc:
            print(f"[voice] 语音命令监听停止：{exc}")


def encode_frame_to_image_url(
    frame: np.ndarray,
    max_width: int,
    jpeg_quality: int,
    image_mode: str,
) -> str:
    require_cv2()
    if max_width > 0 and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        target_size = (max_width, max(1, int(frame.shape[0] * scale)))
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode frame as JPEG")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    if image_mode == "data_uri":
        return f"data:image/jpeg;base64,{b64}"
    if image_mode == "raw_base64":
        return b64
    raise ValueError(f"unsupported image mode: {image_mode}")


def list_cameras(max_index: int = 6) -> None:
    require_cv2()
    print("可用摄像头探测结果：")
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        width = height = 0
        if ok:
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
        cap.release()
        status = "OK" if ok else "不可用"
        detail = f" {width}x{height}" if width and height else ""
        print(f"  {index}: {status}{detail}")


def build_client(args: argparse.Namespace) -> Glm4vClient | DryRunClient:
    if args.dry_run:
        return DryRunClient()
    api_key = args.api_key or os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 API Key。请设置环境变量 ZHIPUAI_API_KEY，或通过 --api-key 传入。"
        )
    return Glm4vClient(
        api_key=api_key,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def build_assistant(args: argparse.Namespace) -> RehabAssistant:
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    return RehabAssistant(
        client=build_client(args),
        exercise_name=args.exercise_name,
        target_reps=args.target_reps,
        weight_kg=args.weight_kg,
        image_mode=args.image_mode,
        jpeg_quality=args.jpeg_quality,
        image_max_width=args.image_max_width,
        system_prompt=system_prompt,
    )


def handle_voice_text(
    text: str,
    assistant: RehabAssistant,
    latest_frame: np.ndarray | None,
) -> str:
    stop_words = ("退出程序", "停止程序", "关闭程序", "quit", "exit")
    summary_words = ("总结", "结束本组", "完成本组", "卡路里", "热量", "消耗")
    if contains_any(text, stop_words):
        return "stop"
    if contains_any(text, summary_words):
        assistant.summarize_set(text)
        return "summary"
    assistant.answer_voice_question(text, latest_frame)
    return "question"


def run_image_once(args: argparse.Namespace) -> int:
    require_cv2()
    assistant = build_assistant(args)
    frame = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"无法读取图片：{args.image_file}")
    keyframe = KeyFrame(
        frame=frame,
        motion_score=0.0,
        displacement_score=0.0,
        duration_seconds=0.0,
        reason="image_file",
    )
    assistant.analyze_keyframe(keyframe, voice_text=args.voice_text)
    if args.target_reps <= 1:
        assistant.summarize_set()
    return 0


def run_camera_loop(args: argparse.Namespace) -> int:
    require_cv2()
    assistant = build_assistant(args)
    extractor = MotionKeyFrameExtractor(
        motion_threshold=args.motion_threshold,
        stable_ratio=args.stable_ratio,
        stable_frames=args.stable_frames,
        cooldown_frames=args.cooldown_frames,
        min_cycle_seconds=args.min_cycle_seconds,
        max_cycle_seconds=args.max_cycle_seconds,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={args.camera_index}，可先运行 --list-cameras。")
    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    voice_queue: queue.Queue[str] = queue.Queue()
    voice_worker: VoiceCommandWorker | None = None
    if args.enable_voice:
        voice_worker = VoiceCommandWorker(
            output_queue=voice_queue,
            duration=args.voice_duration,
            interval=args.voice_interval,
            sample_rate=args.voice_sample_rate,
            input_device=args.input_device,
            asr_backend=args.asr_backend,
            language=args.language,
            whisper_model=args.whisper_model,
        )
        voice_worker.start()

    print("摄像头康复动作分析已启动。")
    print("按键：space=手动发送当前帧，s=总结本组，q=退出。Ctrl+C 也可退出。")
    if args.dry_run:
        print("当前为 dry-run，不会真实请求 GLM API。")

    latest_frame: np.ndarray | None = None
    latest_motion = 0.0
    latest_displacement = 0.0
    next_process_at = 0.0
    running = True

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头失败，准备退出。")
                break
            latest_frame = frame.copy()
            now = time.monotonic()

            while not voice_queue.empty():
                action = handle_voice_text(voice_queue.get_nowait(), assistant, latest_frame)
                if action == "stop":
                    running = False
                    break
            if not running:
                break

            if now >= next_process_at:
                keyframe, latest_motion, latest_displacement = extractor.update(frame, now)
                next_process_at = now + 1.0 / args.sample_fps
                if keyframe is not None:
                    assistant.analyze_keyframe(keyframe)
                    if assistant.rep_count >= args.target_reps:
                        assistant.summarize_set()
                        if args.stop_after_set:
                            running = False

            if args.show_preview:
                preview = draw_overlay(
                    frame,
                    rep_count=assistant.rep_count,
                    target_reps=args.target_reps,
                    motion=latest_motion,
                    displacement=latest_displacement,
                    in_motion=extractor.in_motion,
                )
                cv2.imshow("GLM-4V rehab camera", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                elif key == ord("s"):
                    assistant.summarize_set("用户按键触发总结")
                elif key == 32:
                    manual_keyframe = extractor.force_emit(frame, reason="manual_space_key")
                    assistant.analyze_keyframe(manual_keyframe, voice_text="用户按空格手动触发分析。")
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出。")
    finally:
        if voice_worker is not None:
            voice_worker.stop()
        cap.release()
        if args.show_preview:
            cv2.destroyAllWindows()

    if args.summary_on_exit and assistant.rep_count > 0 and not assistant.summary_done:
        assistant.summarize_set("程序退出时自动总结")
    return 0


def draw_overlay(
    frame: np.ndarray,
    rep_count: int,
    target_reps: int,
    motion: float,
    displacement: float,
    in_motion: bool,
) -> np.ndarray:
    preview = frame.copy()
    status = "motion" if in_motion else "stable"
    lines = [
        f"reps: {rep_count}/{target_reps}",
        f"motion: {motion:.2f}",
        f"disp: {displacement:.2f}",
        f"state: {status}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 220, 60),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通过 HTTP API 调用 GLM-4V-Flash，分析摄像头康复体操关键帧。"
    )
    parser.add_argument("--api-key", default=None, help="智谱 API Key；也可用 ZHIPUAI_API_KEY/GLM_API_KEY。")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="GLM Chat Completions endpoint。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称，默认 glm-4v-flash。")
    parser.add_argument("--timeout", type=positive_float, default=60.0, help="API 请求超时秒数。")
    parser.add_argument("--max-retries", type=int, default=2, help="API 请求失败重试次数。")
    parser.add_argument("--temperature", type=float, default=0.2, help="模型温度。")
    parser.add_argument("--max-tokens", type=positive_int, default=768, help="模型最大输出 token。")
    parser.add_argument("--dry-run", action="store_true", help="只构造请求并打印模拟回复，不访问网络。")

    parser.add_argument("--exercise-name", default="缓慢高抬腿", help="当前康复体操动作名称。")
    parser.add_argument("--target-reps", type=positive_int, default=8, help="一组目标动作次数。")
    parser.add_argument("--weight-kg", type=positive_float, default=60.0, help="用户体重，用于卡路里估计。")
    parser.add_argument("--system-prompt-file", default=None, help="自定义系统提示词文件。")
    parser.add_argument("--voice-text", default="", help="--image-file 单张测试时附加的用户语音文本。")

    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV 摄像头编号。")
    parser.add_argument("--camera-width", type=int, default=640, help="摄像头采集宽度；0 表示不设置。")
    parser.add_argument("--camera-height", type=int, default=480, help="摄像头采集高度；0 表示不设置。")
    parser.add_argument("--sample-fps", type=positive_float, default=8.0, help="关键帧检测采样帧率。")
    parser.add_argument("--show-preview", action="store_true", help="显示摄像头预览窗口和按键控制。")
    parser.add_argument("--list-cameras", action="store_true", help="探测摄像头编号后退出。")
    parser.add_argument("--image-file", default=None, help="不用摄像头，直接分析本地图片。")

    parser.add_argument("--image-mode", choices=["data_uri", "raw_base64"], default="data_uri")
    parser.add_argument("--image-max-width", type=positive_int, default=640, help="发送给 API 前缩放到的最大宽度。")
    parser.add_argument("--jpeg-quality", type=positive_int, default=82, help="JPEG 压缩质量，1-100。")

    parser.add_argument("--motion-threshold", type=positive_float, default=6.0, help="动作开始的帧差阈值。")
    parser.add_argument("--stable-ratio", type=positive_float, default=0.55, help="稳定阈值相对 motion-threshold 的比例。")
    parser.add_argument("--stable-frames", type=positive_int, default=5, help="连续稳定多少帧后发送关键帧。")
    parser.add_argument("--cooldown-frames", type=int, default=6, help="关键帧发送后的冷却帧数。")
    parser.add_argument("--min-cycle-seconds", type=positive_float, default=0.7, help="最短动作循环秒数。")
    parser.add_argument("--max-cycle-seconds", type=positive_float, default=6.0, help="最长动作循环秒数，超时强制发送。")
    parser.add_argument("--stop-after-set", action="store_true", help="达到目标次数并总结后退出。")
    parser.add_argument("--summary-on-exit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--enable-voice", action="store_true", help="开启麦克风语音命令输入。")
    parser.add_argument("--voice-duration", type=positive_float, default=3.0, help="每次语音监听录音秒数。")
    parser.add_argument("--voice-interval", type=positive_float, default=7.0, help="语音监听间隔秒数。")
    parser.add_argument("--voice-sample-rate", type=positive_int, default=16000, help="语音采样率。")
    parser.add_argument("--input-device", default=None, help="sounddevice 输入设备编号或名称。")
    parser.add_argument(
        "--asr-backend",
        choices=["google", "sphinx", "whisper", "faster-whisper"],
        default="faster-whisper",
        help="语音交互 ASR 后端。",
    )
    parser.add_argument("--language", choices=["zh-CN", "en-US"], default="zh-CN", help="ASR 语言。")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper/faster-whisper 模型大小或路径。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.jpeg_quality > 100:
        raise argparse.ArgumentTypeError("--jpeg-quality must be <= 100")
    if args.list_cameras:
        list_cameras()
        return 0
    if args.image_file:
        return run_image_once(args)
    return run_camera_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
