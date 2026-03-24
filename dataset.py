from pydantic import BaseModel
import copy

import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

class ESC50DatasetParams(BaseModel):
    metadata_paths: list[str]

class ESC50Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        target = self.df.iloc[item]

        inputs, sr = torchaudio.load_with_torchcodec(target["path"].replace("ssd/data", "storage/audio-encoder-project"))
        input_length = inputs.size(-1)
        label = target["target"]

        return inputs, input_length, label

def get_esc50_dataset(dp: ESC50DatasetParams, target_index: int):
    train_df = copy.deepcopy(dp.metadata_paths)
    eval_df = train_df.pop(target_index)

    train_df_list = [pd.read_csv(path) for path in train_df]
    train_df = pd.concat(train_df_list, ignore_index=True)
    train_dataset = ESC50Dataset(train_df)

    eval_df = pd.read_csv(eval_df)
    eval_dataset = ESC50Dataset(eval_df)

    return train_dataset, eval_dataset


class DatasetParams(BaseModel):
    train_metadata_path: str
    eval_metadata_path: str
    test_metadata_path: str


class CustomizedDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int):
        target = self.df.iloc[item]

        try:
            inputs, sr = torchaudio.load_with_torchcodec(target["path"])
        except:
            print(target["path"])
            print("error")
            return None
        input_length = inputs.size(-1)
        inputs = inputs.mean(dim=0, keepdim=True)
        label = target["target"]

        max_length = sr * 30
        if input_length > max_length:
            input_length = max_length
            inputs = inputs[:, :max_length]

        return inputs, input_length, label

def get_dataset(dp: DatasetParams):
    train_df = CustomizedDataset(pd.read_csv(dp.train_metadata_path))
    eval_df = CustomizedDataset(pd.read_csv(dp.eval_metadata_path))
    test_df = CustomizedDataset(pd.read_csv(dp.test_metadata_path))

    return train_df, eval_df, test_df

def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    audio = pad_sequence([item[0].transpose(0, 1) for item in batch], batch_first=True, padding_value=0.0)
    audio = audio.transpose(1, 2)
    return audio, torch.tensor([item[1] for item in batch], dtype=torch.long), torch.tensor([item[2] for item in batch], dtype=torch.long)
