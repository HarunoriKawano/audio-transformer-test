import time
import numpy as np
import torch
from vision_transformer import ViT
from model_for_test import Model, Config
import torchaudio
from audioencoder import Config as EncoderConfig
from utils.others.json_to_instance import json_to_instance
import tqdm

# 論文用の公平な比較のため、CPUのスレッド数を1に固定（必須）
torch.set_num_threads(1)

def measure_cpu_latency_and_rtf(model, inputs, audio_duration_sec, num_warmup=10, num_runs=50):
    """
    CPU環境での推論レイテンシとRTFを計測する関数

    Args:
        model: 評価対象のPyTorchモデル
        inputs: モデルに入力するダミーテンソル
        audio_duration_sec (float): 入力データの実際の音声長（秒）
        num_warmup (int): 計測から除外するウォームアップ回数
        num_runs (int): レイテンシを計算するための本番の推論回数

    Returns:
        mean_latency_ms (float): 平均レイテンシ（ミリ秒）
        std_latency_ms (float): レイテンシの標準偏差（ミリ秒）
        rtf (float): リアルタイム係数 (Real-Time Factor)
    """
    model.eval()

    with torch.no_grad():
        # 1. ウォームアップ（CPUキャッシュの初期化オーバーヘッドを除外するため）
        for _ in tqdm.tqdm(range(num_warmup)):
            _ = model(*inputs)

        # 2. 本計測
        latencies = []
        for _ in tqdm.tqdm(range(num_runs)):
            start_time = time.perf_counter() # 高精度タイマー
            _ = model(*inputs)
            end_time = time.perf_counter()

            # 秒からミリ秒に変換して記録
            latencies.append((end_time - start_time) * 1000)

    # 3. 統計値とRTFの計算
    mean_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)

    # RTF = 推論にかかった時間(秒) / 実際の音声長(秒)
    mean_latency_sec = mean_latency_ms / 1000.0
    rtf = mean_latency_sec / audio_duration_sec

    return mean_latency_ms, std_latency_ms, rtf


# ==========================================
# 実行例：wav2vec 2.0 (Base) で 10秒の音声をテスト
# ==========================================
if __name__ == "__main__":
    print("計測を準備中...")

    wav2vec2 = torchaudio.models.wav2vec2_base().eval()
    vit = ViT(768, 8, 12).eval()
    save_dir = "save_dir/finetuning/gsc_v1"
    config_path = f"{save_dir}/config.json"
    model_config_path = f"{save_dir}/encoder_config.json"
    config = json_to_instance(config_path, Config)
    encoder_config = json_to_instance(model_config_path, EncoderConfig)
    model = Model(encoder_config, config).eval()

    duration = 10

    dummy = torch.rand(1, 1, 16000 * duration)
    input_lengths = torch.tensor([16000 * duration])

    mean_ms, std_ms, rtf = measure_cpu_latency_and_rtf(
        model=model,
        inputs=(dummy, input_lengths),
        audio_duration_sec=duration,
        num_warmup=10,
        num_runs=30 # 論文用には50〜100回程度回すと値が安定します
    )

    print(f"Average Latency : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Real-Time Factor: {rtf:.3f}")

    if rtf < 1.0:
        print("=> RTFが1.0未満のため、リアルタイム処理が可能です！")
    else:
        print("=> RTFが1.0以上のため、実際の時間より処理に時間がかかっています。")