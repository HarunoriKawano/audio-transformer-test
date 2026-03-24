import time
import numpy as np
import torch
from vision_transformer import ViT
from model_for_test import Model, Config
import torchaudio
from audioencoder import Config as EncoderConfig
from utils.others.json_to_instance import json_to_instance
import tqdm
import resource

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

        # ==========================================
    # 2. ウォームアップ（CPUのキャッシュを安定させる）
    # ==========================================
    print("ウォームアップを実行中...")
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_warmup)):
            _ = model(*inputs)

    # ==========================================
    # 3. 本計測（RTF と ピークメモリ）
    # ==========================================
    print("推論速度（RTF）およびピークメモリを計測中...")
    total_time = 0.0

    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_runs)):
            start_time = time.time()
            _ = model(*inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

    # --- ピークメモリの取得 ---
    # Linux環境(GCP Ubuntu等)では ru_maxrss はキロバイト(KB)単位で返ってきます
    # MB（メガバイト）に変換するために 1024 で割ります
    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_mb = peak_memory_kb / 1024.0

    # --- RTFの計算 ---
    avg_inference_time = total_time / num_runs
    rtf = avg_inference_time / audio_duration_sec

    # ==========================================
    # 4. 結果出力
    # ==========================================
    print("\n" + "="*20 + " 最終結果 " + "="*20)
    print(f"Average Inference Time : {avg_inference_time:.4f} seconds")
    print(f"RTF (Real-Time Factor) : {rtf:.4f}")
    print(f"Peak Memory Footprint  : {peak_memory_mb:.2f} MB")
    print("="*50)


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

    dummy = torch.rand(1,1,  16000 * duration)
    input_lengths = torch.tensor([16000 * duration])

    measure_cpu_latency_and_rtf(
        model=model,
        inputs=(dummy, input_lengths),
        audio_duration_sec=duration,
        num_warmup=10,
        num_runs=100 # 論文用には50〜100回程度回すと値が安定します
    )