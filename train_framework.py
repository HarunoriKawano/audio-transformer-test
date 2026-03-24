import os
from statistics import mean
import json

from model_with_spec import Model
from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.cuda.amp import GradScaler, autocast

from dataset import collate_fn
from utils.scheduler.cosine_decay_scheduler import get_cosine_scheduler
from audioencoder import Preprocessor

class LearningSettings(BaseModel):
    batch_size: int
    max_epoch_num: int
    learning_rate: float
    mixed_precision: bool
    cpu_num_works: int = 4


class TrainFramework:
    bar_format = '{n_fmt}/{total_fmt}: {percentage:3.0f}%, [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    def __init__(self, model: Model, ls: LearningSettings, train_dataset: Dataset, eval_dataset: Dataset):
        self.ls = ls

        self.model = model.cuda()
        self.best_model_params = self.model.state_dict()
        self.model.eval()
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        """
        self.criterion = nn.CrossEntropyLoss()

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.ls.batch_size, num_workers=self.ls.cpu_num_works,
            pin_memory=True, collate_fn=collate_fn
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, batch_size=self.ls.batch_size, num_workers=self.ls.cpu_num_works,
            pin_memory=True, collate_fn=collate_fn
        )

        self._num_total_steps = len(self.train_dataloader) * self.ls.max_epoch_num
        self._num_params: int = 0
        for p in self.model.parameters():
            self._num_params += p.numel()
        """
        for param in self.model.encoder.encoder.parameters():
            param.requires_grad = False
        """

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.ls.learning_rate)
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            self._num_total_steps // 10,
            self._num_total_steps
        )
        self.scaler: GradScaler = GradScaler()


    def train(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        best_acc = 0.0
        best_epoch = 0

        self._strong_print([
            "Training Start",
            f"Max epoch: {self.ls.max_epoch_num}",
            f"Total step: {self._num_total_steps}",
            f"Model size: {self._num_params}"
        ])

        for epoch in range(self.ls.max_epoch_num):
            epoch += 1
            train_loss_list: list[float] = []
            eval_loss_list: list[float] = []
            eval_predicted_list: list[torch.Tensor] = []
            eval_labels_list: list[torch.Tensor] = []

            # Train step
            self.model.train()
            with tqdm(self.train_dataloader, bar_format=self.bar_format) as pbar:
                pbar.set_description(f'[Train] [Epoch {epoch}/{self.ls.max_epoch_num}]')
                for data in pbar:
                    loss = self._train_step(*data)
                    train_loss_list.append(loss)
                    pbar.set_postfix({"Loss": loss})


            # Evaluation step
            self.model.eval()
            with tqdm(self.eval_dataloader, bar_format=self.bar_format) as pbar:
                pbar.set_description(f'[Eval] [Epoch {epoch}/{self.ls.max_epoch_num}]')
                for data in pbar:
                    loss, predicted, labels = self._eval_step(*data)
                    eval_loss_list.append(loss)
                    eval_predicted_list.append(predicted)
                    eval_labels_list.append(labels)
                    pbar.set_postfix({"Loss": loss})

            # Log
            train_loss = mean(train_loss_list)
            eval_loss = mean(eval_loss_list)
            eval_accuracy = self._evaluation(eval_predicted_list, eval_labels_list)

            epoch_log_df = pd.DataFrame([{
                "epoch": epoch,
                "step": len(self.train_dataloader) * epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy
            }])
            epoch_log_df.to_csv(
                os.path.join(save_dir, "epoch_log.csv"),
                encoding="utf-8",
                mode="w" if epoch == 1 else "a",
                index=False,
                header=epoch == 1
            )

            self._strong_print([
                f"Epoch: {epoch}",
                f"Train loss: {train_loss}",
                f"Eval loss: {eval_loss}",
                f"Accuracy: {eval_accuracy}"
            ])

            # save model
            self.model.cpu()
            torch.save(self.model.state_dict(), os.path.join(save_dir, "last_model.pth"))
            if best_acc < eval_accuracy:
                best_acc = eval_accuracy
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            self.model.cuda()

        result_dict = {
            "Total_epoch": self.ls.max_epoch_num,
            "Total_step": self.ls.max_epoch_num * len(self.train_dataloader),
            "Best_epoch": best_epoch,
            "Best_val_accuracy": best_acc,
            "Num params": self._num_params
        }
        with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(result_dict, f)

        return best_acc


    def test(self, test_dataset: Dataset):
        test_loss_list: list[float] = []
        test_predicted_list: list[torch.Tensor] = []
        test_labels_list: list[torch.Tensor] = []

        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=self.ls.batch_size, num_workers=self.ls.cpu_num_works,
            pin_memory=True, persistent_workers=True, collate_fn=collate_fn
        )

        self.model.eval()
        with tqdm(test_dataloader, bar_format=self.bar_format) as pbar:
            pbar.set_description(f'[Test]')
            for data in pbar:
                loss, predicted, labels = self._eval_step(*data)
                test_loss_list.append(loss)
                test_predicted_list.append(predicted)
                test_labels_list.append(labels)
                pbar.set_postfix({"Loss": loss})

        test_loss = mean(test_loss_list)
        accuracy = self._evaluation(test_predicted_list, test_labels_list)

        return test_loss, accuracy


    def _train_step(self, inputs, input_lengths, labels):
        self.optimizer.zero_grad()
        inputs = inputs.cuda(non_blocking=True)
        input_lengths = input_lengths.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with autocast(enabled=self.ls.mixed_precision, dtype=torch.bfloat16):
            out = self.model(inputs, input_lengths)
            loss = self.criterion(out, labels)

        if self.ls.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        return loss.item()


    def _eval_step(self, inputs, input_lengths, labels):
        inputs = inputs.cuda(non_blocking=True)
        input_lengths = input_lengths.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.no_grad():
            out = self.model(inputs, input_lengths)
            loss = self.criterion(out, labels)
            predicted = torch.argmax(out, dim=-1)

        predicted = predicted.detach().cpu()
        labels = labels.detach().cpu()

        return loss.item(), predicted, labels

    @staticmethod
    def _evaluation(predicted_list, labels_list):
        predicted = torch.cat(predicted_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        accuracy = (predicted == labels).sum() / len(labels)

        return accuracy.item()


    @staticmethod
    def _strong_print(strings: list[str]):
        max_length = max([len(string) for string in strings])
        print()
        print("=" * (max_length + 4))
        for string in strings:
            print(f" {string} ")
        print("=" * (max_length + 4))
        print()

