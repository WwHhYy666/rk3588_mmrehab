import pyttsx3
import time


def main():
    print("正在初始化 TTS 语音引擎...")

    engine = pyttsx3.init()

    rate = engine.getProperty("rate")
    volume = engine.getProperty("volume")
    voices = engine.getProperty("voices")

    print("当前语速：", rate)
    print("当前音量：", volume)
    print("可用语音数量：", len(voices))

    print()
    print("系统可用语音列表：")
    for index, voice in enumerate(voices):
        print(index, voice.name, voice.id)

    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)

    selected_voice = None

    for voice in voices:
        voice_info = (voice.name + " " + voice.id).lower()

        if "chinese" in voice_info or "huihui" in voice_info or "zh" in voice_info:
            selected_voice = voice.id
            break

    if selected_voice is not None:
        engine.setProperty("voice", selected_voice)
        print()
        print("已尝试切换到中文语音：", selected_voice)
    else:
        print()
        print("没有自动找到中文语音，将使用系统默认语音。")

    text = "您好，请慢慢弯曲膝盖，保持动作稳定。"

    print()
    print("准备播放语音...")
    print("播报内容：", text)

    engine.say(text)
    engine.runAndWait()

    time.sleep(0.5)

    print("TTS 测试完成。")


if __name__ == "__main__":
    main()
