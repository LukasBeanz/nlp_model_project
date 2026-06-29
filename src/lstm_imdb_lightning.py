# -*- coding: utf-8 -*-
"""
IMDB 영화 리뷰 감성 분석 LSTM 모델 - PyTorch Lightning 정상 실행 버전
"""

# ---------------------------------------------------------------------
# 1. 기본 라이브러리 불러오기
# ---------------------------------------------------------------------

import os
import re
import tarfile
import random
import urllib.request

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# 2. 딥러닝 라이브러리 불러오기
# ---------------------------------------------------------------------

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy


# ---------------------------------------------------------------------
# 3. 설정값 정의
# ---------------------------------------------------------------------

@dataclass
class Config:
    """프로젝트 전체에서 사용할 설정값을 저장하는 클래스입니다."""

    data_url: str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir: str = "../data"
    archive_name: str = "aclImdb_v1.tar.gz"
    dataset_folder: str = "aclImdb"
    max_len: int = 200
    max_vocab_size: int = 20000
    min_freq: int = 2
    batch_size: int = 64
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    learning_rate: float = 0.001
    max_epochs: int = 3
    val_ratio: float = 0.2
    num_workers: int = 0
    seed: int = 42
    use_toy_data_if_download_fails: bool = True


# ---------------------------------------------------------------------
# 4. 텍스트 전처리 함수
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """영화 리뷰 원문을 모델에 넣기 쉬운 형태로 정리합니다."""

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9!?.,' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()

    return text


def tokenize(text: str) -> List[str]:
    """문장을 단어 리스트로 분리합니다."""

    return clean_text(text).split()


# ---------------------------------------------------------------------
# 5. IMDB 데이터 다운로드 및 로드 함수
# ---------------------------------------------------------------------

def download_and_extract_imdb(config: Config) -> Path:
    """IMDB 데이터셋이 없으면 다운로드하고 압축을 해제합니다."""

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / config.dataset_folder

    if dataset_path.exists():
        print(f"[데이터 확인] 기존 IMDB 데이터셋 사용: {dataset_path}")
        return dataset_path

    archive_path = data_dir / config.archive_name

    if not archive_path.exists():
        print("[데이터 다운로드] IMDB 데이터셋 다운로드를 시작합니다.")
        print(f"[URL] {config.data_url}")
        urllib.request.urlretrieve(config.data_url, archive_path)
        print(f"[데이터 다운로드 완료] {archive_path}")

    print("[압축 해제] IMDB 데이터셋 압축을 해제합니다.")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print(f"[압축 해제 완료] {dataset_path}")
    return dataset_path


def read_imdb_split(dataset_path: Path, split: str) -> List[Tuple[str, int]]:
    """IMDB train 또는 test 폴더에서 리뷰 텍스트와 라벨을 읽어옵니다."""

    samples: List[Tuple[str, int]] = []
    label_map = {"neg": 0, "pos": 1}

    for label_name, label_id in label_map.items():

        review_dir = dataset_path / split / label_name

        if not review_dir.exists():
            raise FileNotFoundError(f"리뷰 폴더를 찾을 수 없습니다: {review_dir}")

        for file_path in sorted(review_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            samples.append((text, label_id))

    random.shuffle(samples)
    return samples


def make_toy_samples() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """인터넷 다운로드가 불가능할 때 실행 확인용 작은 예제 데이터를 만듭니다."""

    positive = [
        "This movie was wonderful and I loved every moment",
        "The story was beautiful and the acting was excellent",
        "A fantastic film with great characters",
        "I really enjoyed this movie it was amazing",
        "The plot was touching and the music was great",
        "Brilliant movie with a very satisfying ending",
        "The performances were strong and emotional",
        "This is one of the best films I have watched",
    ]

    negative = [
        "This movie was terrible and boring",
        "The story was weak and the acting was bad",
        "A disappointing film with poor characters",
        "I did not enjoy this movie it was awful",
        "The plot was confusing and the music was annoying",
        "Bad movie with a very unsatisfying ending",
        "The performances were weak and emotionless",
        "This is one of the worst films I have watched",
    ]

    samples = [(text, 1) for text in positive] + [(text, 0) for text in negative]
    samples = samples * 20
    random.shuffle(samples)

    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


def load_data(config: Config) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """IMDB 데이터를 로드하고, 실패하면 선택적으로 예제 데이터를 반환합니다."""

    try:
        dataset_path = download_and_extract_imdb(config)
        train_samples = read_imdb_split(dataset_path, "train")
        test_samples = read_imdb_split(dataset_path, "test")

        print(f"[데이터 로드 완료] train={len(train_samples)}, test={len(test_samples)}")
        return train_samples, test_samples

    except Exception as error:
        print(f"[경고] IMDB 원본 데이터 로드 실패: {error}")

        if not config.use_toy_data_if_download_fails:
            raise

        print("[대체 실행] 인터넷 다운로드가 불가능하여 작은 예제 데이터로 실행합니다.")
        return make_toy_samples()


# ---------------------------------------------------------------------
# 6. Vocabulary 생성 함수
# ---------------------------------------------------------------------

def build_vocab(samples: List[Tuple[str, int]], config: Config) -> Dict[str, int]:
    """훈련 데이터에서 단어 사전을 만듭니다."""

    counter: Counter = Counter()

    for text, _ in samples:
        counter.update(tokenize(text))

    word_to_index: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    for word, freq in counter.most_common(config.max_vocab_size - len(word_to_index)):

        if freq < config.min_freq:
            continue

        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    print(f"[Vocabulary 생성 완료] 단어 수: {len(word_to_index)}")
    return word_to_index


def encode_text(text: str, word_to_index: Dict[str, int], max_len: int) -> torch.Tensor:
    """문장 하나를 고정 길이 정수 텐서로 변환합니다."""

    tokens = tokenize(text)
    token_ids = [word_to_index.get(token, word_to_index["<unk>"]) for token in tokens]
    token_ids = token_ids[:max_len]

    if len(token_ids) < max_len:
        token_ids = token_ids + [word_to_index["<pad>"]] * (max_len - len(token_ids))

    return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 7. Dataset 클래스 정의
# ---------------------------------------------------------------------

class IMDBDataset(Dataset):
    """IMDB 리뷰 텍스트와 라벨을 PyTorch Dataset 형태로 제공하는 클래스입니다."""

    def __init__(self, samples: List[Tuple[str, int]], word_to_index: Dict[str, int], max_len: int):
        self.samples = samples
        self.word_to_index = word_to_index
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label = self.samples[index]
        input_ids = encode_text(text, self.word_to_index, self.max_len)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, label_tensor


# ---------------------------------------------------------------------
# 8. LightningDataModule 정의
# ---------------------------------------------------------------------

class IMDBDataModule(pl.LightningDataModule):
    """데이터 준비와 DataLoader 생성을 담당하는 Lightning DataModule입니다."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_to_index: Dict[str, int] = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        train_samples, test_samples = load_data(self.config)
        self.word_to_index = build_vocab(train_samples, self.config)

        full_train_dataset = IMDBDataset(train_samples, self.word_to_index, self.config.max_len)
        self.test_dataset = IMDBDataset(test_samples, self.word_to_index, self.config.max_len)

        val_size = int(len(full_train_dataset) * self.config.val_ratio)
        train_size = len(full_train_dataset) - val_size

        generator = torch.Generator().manual_seed(self.config.seed)

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator,
        )

        print(f"[Dataset 준비 완료] train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


# ---------------------------------------------------------------------
# 9. LSTM 모델 정의
# ---------------------------------------------------------------------

class LSTMClassifier(pl.LightningModule):
    """IMDB 리뷰 감성 분석을 위한 LSTM 분류 모델입니다."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        pad_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        sentence_vector = hidden[-1]
        sentence_vector = self.dropout(sentence_vector)
        logits = self.classifier(sentence_vector)

        return logits

    def _shared_step(self, batch, stage: str):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, labels)
        elif stage == "val":
            acc = self.val_acc(preds, labels)
        else:
            acc = self.test_acc(preds, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# ---------------------------------------------------------------------
# 10. 예측 함수
# ---------------------------------------------------------------------

def predict_sentiment(model: LSTMClassifier, text: str, word_to_index: Dict[str, int], config: Config) -> Tuple[str, float]:
    """학습된 모델로 문장 하나의 감성을 예측합니다."""

    model.eval()

    with torch.no_grad():
        input_ids = encode_text(text, word_to_index, config.max_len)
        input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model.device)

        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_id].item()

    label = "positive" if pred_id == 1 else "negative"
    return label, confidence


# ---------------------------------------------------------------------
# 11. main 함수
# ---------------------------------------------------------------------

def main() -> None:
    """전체 실행 흐름을 담당하는 main 함수입니다."""

    config = Config()
    pl.seed_everything(config.seed, workers=True)

    data_module = IMDBDataModule(config)
    data_module.setup(stage="fit")

    vocab_size = len(data_module.word_to_index)

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        pad_index=data_module.word_to_index["<pad>"],
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    examples = [
        "This movie was fantastic and the acting was excellent.",
        "The film was boring and the story was terrible.",
    ]

    print("\n[예측 예시]")
    for text in examples:
        label, confidence = predict_sentiment(model, text, data_module.word_to_index, config)
        print(f"문장: {text}")
        print(f"예측: {label}, 신뢰도: {confidence:.4f}\n")


# ---------------------------------------------------------------------
# 12. 프로그램 시작 지점
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
