# -*- coding: utf-8 -*-
"""
IMDB 영화 리뷰 감성 분류(긍정 pos / 부정 neg) — LSTM + PyTorch Lightning (현대화 버전)

[원본과의 차이]
원본 코드는 torchtext.legacy API(Field, BucketIterator, IMDB)를 사용했으나,
torchtext.legacy 는 최신 torchtext(0.12+)에서 완전히 제거되어 더 이상 동작하지 않습니다.
그래서 이 파일은 다음과 같이 현대적인 방식으로 재작성했습니다.

    - 데이터셋:   torchtext.legacy.datasets.IMDB  ->  HuggingFace datasets 의 load_dataset("imdb")
    - 전처리:     torchtext Field                 ->  직접 구현한 토크나이저 + 단어 사전(vocab)
    - 데이터로더: BucketIterator                  ->  torch.utils.data 의 Dataset + DataLoader
    - 임베딩:     vocab.vectors(FastText) 인덱싱  ->  학습 가능한 nn.Embedding 계층
    - 지표:       pl.metrics.Accuracy            ->  torchmetrics.Accuracy
    - 클래스 수:  output_size=3(legacy <unk> 라벨) ->  output_size=2(pos/neg 그대로 0/1)
    - 실행:       모듈 최상위 실행              ->  main() + if __name__ == "__main__" 가드

[필요 패키지]
    pip install torch pytorch-lightning torchmetrics datasets

[전체 실행 흐름]
    1. 라이브러리 불러오기
    2. 토크나이저 / 단어 사전 / 인코딩 함수 정의
    3. PyTorch Dataset 클래스 정의
    4. PyTorch Lightning 기반 LSTM 모델 정의
    5. main(): 데이터 로드 -> 사전 생성 -> DataLoader -> 모델 생성 -> 학습
"""

# ---------------------------------------------------------------------
# 1. 필요한 라이브러리 불러오기
# ---------------------------------------------------------------------

# 정규식으로 간단한 영어 토큰화를 수행하기 위해 사용합니다.
import re

# 난수 시드를 고정하기 위해 사용합니다.
import random

# 단어 빈도수를 세기 위해 사용합니다.
from collections import Counter

# PyTorch 핵심 라이브러리입니다. 텐서 연산, GPU 사용, 학습 등에 사용됩니다.
import torch

# 신경망 계층(LSTM, Linear 등)을 만들 때 사용하는 모듈입니다.
import torch.nn as nn

# 활성화 함수 등을 함수 형태로 사용합니다. 여기서는 ELU 활성화 함수 F.elu()를 사용합니다.
import torch.nn.functional as F

# 사용자 정의 데이터셋과 미니배치 공급용 데이터 로더를 만들기 위해 사용합니다.
from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning은 학습 루프를 구조적으로 관리해 주는 라이브러리입니다.
import pytorch_lightning as pl

# torchmetrics는 정확도 등 평가 지표를 계산하는 최신 라이브러리입니다.
from torchmetrics import Accuracy

# HuggingFace datasets 는 IMDB 등 표준 데이터셋을 손쉽게 내려받고 다루게 해 줍니다.
from datasets import load_dataset


# ---------------------------------------------------------------------
# 2. 토크나이저 / 단어 사전 / 인코딩 함수
# ---------------------------------------------------------------------

# 난수 시드를 고정하여 실행할 때마다 최대한 비슷한 결과가 나오게 합니다.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# -------------------------------------------------

# 간단한 영어 토큰화를 수행하는 함수입니다.
def simple_tokenize(text: str) -> list[str]:
    # 소문자로 변환합니다.
    text = str(text).lower()

    # 알파벳/숫자/작은따옴표로 이루어진 덩어리만 단어로 추출합니다.
    # 이렇게 하면 HTML 태그(<br />)나 문장 부호를 자연스럽게 걸러낼 수 있습니다.
    tokens = re.findall(r"[a-z0-9']+", text)

    return tokens
# -------------------------------------------------

# 훈련 데이터로부터 단어 사전(word -> index)을 만드는 함수입니다.
def build_vocab(texts, min_freq: int = 2, max_size: int = 20000) -> dict[str, int]:
    # 모든 단어의 등장 횟수를 누적할 Counter 객체를 만듭니다.
    counter = Counter()

    # 각 문장을 토큰화하여 단어 빈도를 누적합니다.
    for text in texts:
        counter.update(simple_tokenize(text))
    # for ----------------

    # 0번은 패딩 토큰, 1번은 미등록 단어(<UNK>) 토큰으로 예약합니다.
    word_to_index = {"<PAD>": 0, "<UNK>": 1}

    # 빈도가 높은 단어부터 최대 max_size 개까지 사전에 추가합니다.
    for word, freq in counter.most_common(max_size):
        # min_freq 이상 등장한 단어만 포함합니다.
        if freq >= min_freq:
            word_to_index[word] = len(word_to_index)
    # for ----------------

    return word_to_index
# -------------------------------------------------

# 한 문장을 정수 인덱스 시퀀스로 바꾸고, 고정 길이로 패딩/절단하는 함수입니다.
def encode_and_pad(text: str, word_to_index: dict[str, int], max_len: int) -> list[int]:
    # 문장을 토큰화합니다.
    tokens = simple_tokenize(text)

    # 각 단어를 사전 번호로 변환하고, 사전에 없으면 <UNK> 번호 1을 사용합니다.
    encoded = [word_to_index.get(token, 1) for token in tokens]

    # max_len 보다 길면 앞에서부터 max_len 까지 자릅니다.
    encoded = encoded[:max_len]

    # max_len 보다 짧으면 뒤쪽을 <PAD> 번호 0으로 채워 길이를 맞춥니다.
    padded = encoded + [0] * (max_len - len(encoded))

    return padded
# -------------------------------------------------


# ---------------------------------------------------------------------
# 3. PyTorch Dataset 클래스 정의
# ---------------------------------------------------------------------

# 리뷰 문장과 라벨을 DataLoader 가 읽을 수 있도록 구성하는 Dataset 클래스입니다.
class IMDBDataset(Dataset):
    # 문장, 라벨, 단어 사전, 최대 길이를 받아 저장합니다.
    def __init__(self, texts, labels, word_to_index: dict[str, int], max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.word_to_index = word_to_index
        self.max_len = max_len

    # 전체 샘플 개수를 반환합니다.
    def __len__(self) -> int:
        return len(self.texts)

    # 특정 인덱스 하나에 해당하는 (입력 텐서, 정답 텐서)를 반환합니다.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 문장을 정수 시퀀스로 변환하고 패딩합니다.
        x = encode_and_pad(self.texts[index], self.word_to_index, self.max_len)

        # 입력은 정수 인덱스 시퀀스이므로 LongTensor 로 변환합니다.
        x_tensor = torch.tensor(x, dtype=torch.long)

        # CrossEntropyLoss 는 정답 라벨로 정수(LongTensor)를 받습니다. (IMDB: 0=neg, 1=pos)
        y_tensor = torch.tensor(self.labels[index], dtype=torch.long)

        return x_tensor, y_tensor
# -------------------------------------------------


# ---------------------------------------------------------------------
# 4. PyTorch Lightning 기반 LSTM 모델 정의
# ---------------------------------------------------------------------

# RNNModel 은 LightningModule 을 상속합니다.
# 모델 구조, 학습 단계, 검증 단계, 최적화 설정을 한 클래스 안에 정리합니다.
class RNNModel(pl.LightningModule):
    # 모델 계층을 초기화합니다.
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        output_size: int = 2,
        lr: float = 1e-3,
    ):
        # 부모 클래스 초기화를 먼저 호출합니다. (Lightning 내부 기능에 필수)
        super().__init__()

        # 하이퍼파라미터를 self.hparams 에 저장하고 체크포인트에도 기록합니다.
        self.save_hyperparameters()

        # 학습 가능한 임베딩 계층입니다. padding_idx=0 은 <PAD> 토큰의 기울기를 0으로 고정합니다.
        # 원본처럼 사전학습 벡터를 쓰지 않고, 데이터로부터 임베딩을 함께 학습합니다.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 계층입니다. batch_first=True 로 두어 (배치, 시퀀스, 특징) 형태를 사용합니다.
        # 이렇게 하면 원본처럼 permute 로 차원을 바꿔 줄 필요가 없습니다.
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # LSTM 의 마지막 은닉 상태(hidden_size)를 클래스 점수(output_size)로 변환하는 출력층입니다.
        self.lin = nn.Linear(hidden_size, output_size)

        # 다중 클래스 분류용 손실 함수입니다. 내부에서 softmax 를 처리하므로 logits 를 그대로 넣습니다.
        self.loss_function = nn.CrossEntropyLoss()

        # 훈련/검증 정확도 계산 객체입니다.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_size)

    # 순전파를 정의합니다. 입력 x 형태는 (배치, 시퀀스 길이) 입니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 임베딩을 적용합니다. -> (배치, 시퀀스, 임베딩차원)
        # nn.Embedding 은 모델 파라미터로 등록되어 model.to(device) 시 함께 이동하므로,
        # 원본의 디바이스 불일치 버그(.to(self.device)를 인덱싱 뒤에 호출하던 문제)가 사라집니다.
        x = self.embedding(x)

        # LSTM 을 통과시킵니다.
        # outputs: 모든 시점의 출력, (h_n, c_n): 마지막 은닉/셀 상태
        outputs, (h_n, c_n) = self.lstm(x)

        # 마지막 층의 마지막 은닉 상태를 문장 전체의 요약 벡터로 사용합니다. -> (배치, hidden_size)
        # (원본은 모든 시점의 점수를 sum 했지만, 마지막 hidden 사용이 분류에서 더 일반적입니다.)
        last_hidden = h_n[-1]

        # ELU 활성화 함수로 비선형성을 추가합니다.
        last_hidden = F.elu(last_hidden)

        # 출력층을 통과시켜 클래스 점수(logits)를 만듭니다. -> (배치, output_size)
        logits = self.lin(last_hidden)

        return logits

    # 훈련 배치 하나에 대한 연산을 정의합니다.
    def training_step(self, batch, batch_idx):
        # DataLoader 가 (입력, 라벨) 튜플을 돌려줍니다.
        x, y = batch

        # 예측 점수를 계산합니다.
        y_hat = self(x)

        # 손실을 계산합니다.
        loss = self.loss_function(y_hat, y)

        # 훈련 정확도를 누적 계산합니다.
        self.train_accuracy(y_hat, y)

        # 손실과 정확도를 진행률 표시줄에 기록합니다.
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)

        # 반환한 loss 로 Lightning 이 역전파와 파라미터 갱신을 수행합니다.
        return loss

    # 검증 배치 하나에 대한 연산을 정의합니다.
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # 예측 점수를 계산합니다.
        y_hat = self(x)

        # 검증 손실을 계산합니다.
        loss = self.loss_function(y_hat, y)

        # 검증 정확도를 누적 계산합니다.
        self.val_accuracy(y_hat, y)

        # 검증 손실과 정확도를 기록합니다.
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    # 최적화 알고리즘을 정의합니다.
    def configure_optimizers(self):
        # Adam 옵티마이저를 사용합니다. 학습률은 생성자에서 받은 값(기본 0.001)을 사용합니다.
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
# -------------------------------------------------


# ---------------------------------------------------------------------
# 5. 전체 실행 흐름
# ---------------------------------------------------------------------

# 데이터 준비, 모델 생성, 학습을 수행하는 메인 함수입니다.
def main() -> None:
    # 난수 시드를 고정합니다.
    set_seed(42)

    # 문장 최대 길이와 미니배치 크기를 설정합니다.
    max_len = 200
    batch_size = 32

    # HuggingFace 에서 IMDB 데이터셋을 내려받습니다.
    # 처음 실행 시 인터넷에서 다운로드하며, 이후에는 캐시를 재사용합니다.
    print("IMDB 데이터셋을 불러오는 중...")
    dataset = load_dataset("imdb")

    # 훈련/테스트 문장과 라벨을 분리합니다. (label: 0=neg, 1=pos)
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # 첫 번째 훈련 샘플을 확인합니다.
    print("\n[훈련 데이터 첫 번째 샘플 라벨]:", train_labels[0], "(0=neg, 1=pos)")
    print("[첫 번째 샘플 앞부분]:", train_texts[0][:100], "...")

    # 훈련 데이터로만 단어 사전을 만듭니다.
    word_to_index = build_vocab(train_texts, min_freq=2, max_size=20000)
    vocab_size = len(word_to_index)
    print("\n[단어 집합(vocabulary) 크기]:", vocab_size)

    # Dataset 객체를 만듭니다.
    train_dataset = IMDBDataset(train_texts, train_labels, word_to_index, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, word_to_index, max_len)

    # DataLoader 를 만듭니다.
    # Windows 에서는 num_workers>0 일 때 프로세스 spawn 이슈가 있을 수 있어 0으로 둡니다.
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 모델을 생성합니다. (output_size=2: 긍정/부정)
    model = RNNModel(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_size=128,
        output_size=2,
        lr=1e-3,
    )
    print(model)

    # Trainer 를 생성합니다. (최신 Lightning 방식: accelerator/devices)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=3,
    )

    # 학습을 시작합니다. (DataLoader 를 fit 에 직접 전달)
    trainer.fit(model, train_loader, val_loader)


# 이 파일을 직접 실행할 때만 main() 을 실행합니다.
# (Windows 에서 DataLoader/Trainer 의 프로세스 spawn 안전성을 위해 반드시 필요합니다.)
if __name__ == "__main__":
    main()
