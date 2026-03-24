import torch
from vision_transformer import ViT
from model_for_test import Model, Config
import torchaudio
from audioencoder import Config as EncoderConfig
from utils.others.json_to_instance import json_to_instance
from fvcore.nn import FlopCountAnalysis, flop_count_table


if __name__ == '__main__':
    wav2vec2 = torchaudio.models.wav2vec2_base().eval()
    vit = ViT(768, 8, 12).eval()
    save_dir = "save_dir/transfer/voxceleb1"
    config_path = f"{save_dir}/config.json"
    model_config_path = f"{save_dir}/encoder_config.json"
    config = json_to_instance(config_path, Config)
    encoder_config = json_to_instance(model_config_path, EncoderConfig)
    model = Model(encoder_config, config).eval()

    dummy = torch.rand(1, 1, 16000 * 10)
    input_lengths = torch.tensor([16000 * 10])

    flops = FlopCountAnalysis(vit, dummy)

    # 4. 結果の出力
    # レイヤー（Acoustic Model, Task Modelごとの）詳細な内訳を表示
    print(flop_count_table(flops))

    # 合計GFLOPsの算出 (1 GFLOPs = 10^9 FLOPs)
    total_gflops = flops.total() / 1e9
    print(f"Total GFLOPs: {total_gflops:.3f} GFLOPs")