# -*- coding: utf-8 -*-
"""
네이버 영화 리뷰(NSMC) 감성 분류(긍정 1 / 부정 0) — LSTM + PyTorch Lightning

[원본(IMDB 버전)과의 차이]
영어 IMDB 데이터 대신 한국어 NSMC(Naver Sentiment Movie Corpus)로 감성 분석을 수행합니다.

    - 데이터셋:   load_dataset("imdb")            ->  NSMC 원본 파일(ratings_train/test.txt) 직접 다운로드
    - 토크나이저: 영어 정규식 + 소문자화          ->  한국어 공백 기반 토큰화(한글/영문/숫자만 남김)
    - 결측치:     없음                            ->  빈 리뷰/NaN/중복 제거
    - 평가:       학습 중 검증만                  ->  test_step + trainer.test() 로 최종 평가까지 수행
    - 라벨:       0=neg, 1=pos                    ->  0=부정, 1=긍정 (구조 동일)

[필요 패키지]
    pip install torch pytorch-lightning torchmetrics pandas

[전체 실행 흐름]
    1. 라이브러리 불러오기
    2. NSMC 다운로드 / 한국어 토크나이저 / 단어 사전 / 인코딩 함수 정의
    3. PyTorch Dataset 클래스 정의
    4. PyTorch Lightning 기반 LSTM 모델 정의 (학습/검증/평가)
    5. main(): 데이터 로드 -> 정제 -> 사전 생성 -> DataLoader -> 모델 생성 -> 학습 -> 평가
"""

# ---------------------------------------------------------------------
# 1. 필요한 라이브러리 불러오기
# ---------------------------------------------------------------------

# 파일 경로 처리 및 다운로드 파일 존재 여부 확인에 사용합니다.
import os

# 정규식으로 한국어 토큰화 전처리를 하기 위해 사용합니다.
import re

# 난수 시드를 고정하기 위해 사용합니다.
import random

# 인터넷에서 NSMC 데이터 파일을 내려받기 위해 사용합니다.
import urllib.request

# 단어 빈도수를 세기 위해 사용합니다.
from collections import Counter

# NSMC 는 탭으로 구분된 텍스트 파일이라 pandas 로 읽고 정제합니다.
import pandas as pd

# 통합 데이터(ratings.txt)를 훈련/테스트로 나누기 위해 사용합니다.
from sklearn.model_selection import train_test_split

# PyTorch 핵심 라이브러리입니다.
import torch

# 신경망 계층(LSTM, Linear 등)을 만들 때 사용하는 모듈입니다.
import torch.nn as nn

# 활성화 함수를 함수 형태로 사용합니다. 여기서는 ELU 를 사용합니다.
import torch.nn.functional as F

# 사용자 정의 데이터셋과 미니배치 공급용 데이터 로더를 만들기 위해 사용합니다.
from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning 은 학습 루프를 구조적으로 관리해 주는 라이브러리입니다.
import pytorch_lightning as pl

# torchmetrics 는 정확도 등 평가 지표를 계산하는 라이브러리입니다.
from torchmetrics import Accuracy


# ---------------------------------------------------------------------
# 2. 데이터 다운로드 / 토크나이저 / 단어 사전 / 인코딩 함수
# ---------------------------------------------------------------------

# 난수 시드를 고정하여 실행할 때마다 최대한 비슷한 결과가 나오게 합니다.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# -------------------------------------------------

# NSMC 데이터 파일을 pandas DataFrame 으로 읽고 정제하는 함수입니다.
# 파일이 로컬에 없으면 url 에서 내려받습니다. (이미 data 폴더에 있으면 그대로 사용)
def load_nsmc(path: str, url: str | None = None) -> pd.DataFrame:
    # 저장할 디렉토리가 없으면 만듭니다.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 파일이 없고 url 이 주어졌으면 인터넷에서 내려받습니다.
    if not os.path.exists(path):
        if url is None:
            raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
        print(f"다운로드 중: {url}")
        urllib.request.urlretrieve(url, filename=path)

    # 탭(\t)으로 구분된 파일을 읽습니다. 컬럼: id, document, label
    data = pd.read_csv(path, sep="\t")

    # document 가 비어 있는(NaN) 행을 제거합니다.
    data = data.dropna(subset=["document"])

    # 중복된 리뷰 문장을 하나만 남기고 제거합니다.
    data = data.drop_duplicates(subset=["document"]).reset_index(drop=True)

    return data
# -------------------------------------------------

# 한국어 공백 기반 토큰화를 수행하는 함수입니다.
def simple_tokenize(text: str) -> list[str]:
    # 문자열로 변환합니다.
    text = str(text)

    # 한글(가-힣), 영문, 숫자, 공백만 남기고 나머지(문장부호/이모지 등)는 공백으로 바꿉니다.
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)

    # 공백 기준으로 단어를 나눕니다. (형태소 분석기를 쓰지 않는 간단 방식)
    tokens = text.split()

    return tokens
# -------------------------------------------------

# 훈련 데이터로부터 단어 사전(word -> index)을 만드는 함수입니다.
def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict[str, int]:
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
class NSMCDataset(Dataset):
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

        # CrossEntropyLoss 는 정답 라벨로 정수(LongTensor)를 받습니다. (NSMC: 0=부정, 1=긍정)
        y_tensor = torch.tensor(self.labels[index], dtype=torch.long)

        return x_tensor, y_tensor
# -------------------------------------------------


# ---------------------------------------------------------------------
# 4. PyTorch Lightning 기반 LSTM 모델 정의
# ---------------------------------------------------------------------

# RNNModel 은 LightningModule 을 상속합니다.
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

        # 하이퍼파라미터를 self.hparams 에 저장합니다.
        self.save_hyperparameters()

        # 학습 가능한 임베딩 계층입니다. padding_idx=0 은 <PAD> 토큰의 기울기를 0으로 고정합니다.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 계층입니다. batch_first=True 로 (배치, 시퀀스, 특징) 형태를 사용합니다.
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # LSTM 마지막 은닉 상태를 클래스 점수로 변환하는 출력층입니다.
        self.lin = nn.Linear(hidden_size, output_size)

        # 다중 클래스 분류용 손실 함수입니다. (내부에서 softmax 처리 → logits 그대로 입력)
        self.loss_function = nn.CrossEntropyLoss()

        # 훈련/검증/평가 정확도 계산 객체입니다.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_size)

    # 순전파를 정의합니다. 입력 x 형태는 (배치, 시퀀스 길이) 입니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 임베딩을 적용합니다. -> (배치, 시퀀스, 임베딩차원)
        x = self.embedding(x)

        # LSTM 을 통과시킵니다. (h_n: 마지막 은닉 상태)
        outputs, (h_n, c_n) = self.lstm(x)

        # 마지막 층의 마지막 은닉 상태를 문장 요약 벡터로 사용합니다. -> (배치, hidden_size)
        last_hidden = h_n[-1]

        # ELU 활성화 함수로 비선형성을 추가합니다.
        last_hidden = F.elu(last_hidden)

        # 출력층을 통과시켜 클래스 점수(logits)를 만듭니다. -> (배치, output_size)
        logits = self.lin(last_hidden)

        return logits

    # 훈련 배치 하나에 대한 연산을 정의합니다.
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    # 검증 배치 하나에 대한 연산을 정의합니다.
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        return loss

    # 평가(테스트) 배치 하나에 대한 연산을 정의합니다.
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.test_accuracy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        return loss

    # 최적화 알고리즘을 정의합니다.
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
# -------------------------------------------------


# ---------------------------------------------------------------------
# 5. 전체 실행 흐름
# ---------------------------------------------------------------------

# NSMC 통합 데이터 파일 (data 폴더에 이미 있으면 그대로 사용, 없으면 아래 URL 에서 다운로드)
DATA_PATH = "./data/ratings.txt"
DATA_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt"

# 데이터 준비, 모델 생성, 학습, 평가를 수행하는 메인 함수입니다.
def main() -> None:
    # 난수 시드를 고정합니다.
    set_seed(42)

    # 문장 최대 길이와 미니배치 크기를 설정합니다. (NSMC 리뷰는 짧아 max_len 을 작게 둡니다.)
    max_len = 40
    batch_size = 64

    # NSMC 통합 데이터를 읽고 정제한 뒤, 훈련/테스트로 8:2 분할합니다.
    print("NSMC 데이터셋을 불러오는 중...")
    data = load_nsmc(DATA_PATH, DATA_URL)
    print("전체 샘플 수(정제 후):", len(data))

    # 라벨 비율(긍정/부정)을 유지하며 훈련 80% / 테스트 20% 로 나눕니다.
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    print("훈련 샘플 수:", len(train_df), "/ 테스트 샘플 수:", len(test_df))

    # 문장(document)과 라벨(label)을 분리합니다. (label: 0=부정, 1=긍정)
    train_texts = train_df["document"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["document"].tolist()
    test_labels = test_df["label"].tolist()

    # 첫 번째 훈련 샘플을 확인합니다.
    print("\n[첫 번째 훈련 샘플]:", train_texts[0])
    print("[라벨]:", train_labels[0], "(0=부정, 1=긍정)")

    # 훈련 데이터로만 단어 사전을 만듭니다.
    word_to_index = build_vocab(train_texts, min_freq=2, max_size=30000)
    vocab_size = len(word_to_index)
    print("\n[단어 집합(vocabulary) 크기]:", vocab_size)

    # Dataset 객체를 만듭니다.
    train_dataset = NSMCDataset(train_texts, train_labels, word_to_index, max_len)
    test_dataset = NSMCDataset(test_texts, test_labels, word_to_index, max_len)

    # DataLoader 를 만듭니다. (Windows 안전을 위해 num_workers=0)
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
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

    # Trainer 를 생성합니다. (최신 Lightning 방식)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=5,
    )

    # 학습을 시작합니다. (테스트셋을 검증셋으로 사용해 epoch 별 성능을 확인)
    trainer.fit(model, train_loader, test_loader)

    # 학습이 끝난 뒤 테스트셋으로 최종 평가를 수행합니다.
    print("\n=== 최종 평가 ===")
    trainer.test(model, test_loader)


# 이 파일을 직접 실행할 때만 main() 을 실행합니다.
if __name__ == "__main__":
    main()
