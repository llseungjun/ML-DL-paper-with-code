import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from pathlib import Path

from model import Encoder, Decoder, Seq2Seq
from utils import set_seed, create_dataloaders
from config import Config

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # output: [batch_size, trg_len, vocab_size]
        # trg: [batch_size, trg_len]
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            
            output = model(src, trg, 0)  # teacher forcing 비활성화
            
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def main():
    # 설정 초기화
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # MLflow 실험 설정
    mlflow.set_experiment("seq2seq_translation")
    
    # MLflow run 시작
    with mlflow.start_run():
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            "vocab_size": config.vocab_size,
            "max_len": config.max_len,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs
        })
        
        # 데이터로더 생성
        train_dataloader, valid_dataloader, src_vocab, trg_vocab = create_dataloaders(config)
        
        # 모델 초기화
        encoder = Encoder(len(src_vocab.word2idx), config.hidden_size, config.num_layers, config.dropout)
        decoder = Decoder(len(trg_vocab.word2idx), config.hidden_size, config.num_layers, config.dropout)
        
        model = Seq2Seq(encoder, decoder).to(device)
        
        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0은 패딩 토큰
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        
        # 학습 루프
        best_valid_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
            valid_loss = evaluate(model, valid_dataloader, criterion, device)
            
            # MLflow에 메트릭 로깅
            mlflow.log_metrics({
                "train_loss": train_loss,
                "valid_loss": valid_loss
            }, step=epoch)
            
            print(f'Epoch: {epoch+1}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tValid Loss: {valid_loss:.3f}')
            
            # 최고 성능 모델 저장
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # MLflow에 모델 저장
                mlflow.pytorch.log_model(model, "model")
                # 로컬에도 저장
                save_dir = Path(config.save_dir)
                save_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / 'best-model.pt')

if __name__ == '__main__':
    main() 