import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json
from pathlib import Path

class Vocabulary:
    """어휘 사전을 관리하는 클래스"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        # 특수 토큰 추가
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        self.word2idx[self.sos_token] = 2
        self.word2idx[self.eos_token] = 3
        
        self.idx2word[0] = self.pad_token
        self.idx2word[1] = self.unk_token
        self.idx2word[2] = self.sos_token
        self.idx2word[3] = self.eos_token
        
        self.pad_idx = self.word2idx[self.pad_token]
        self.unk_idx = self.word2idx[self.unk_token]
        self.sos_idx = self.word2idx[self.sos_token]
        self.eos_idx = self.word2idx[self.eos_token]
    
    def build_vocab(self, sentences: List[str], max_vocab_size: int):
        """문장 리스트로부터 어휘 사전을 구축합니다."""
        word_freq = {}
        for sentence in sentences:
            for word in sentence.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도수 기준으로 정렬
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 특수 토큰을 제외한 최대 어휘 수만큼 추가
        for word, _ in sorted_words[:max_vocab_size-4]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, sentence: str) -> List[int]:
        """문장을 토큰 ID 리스트로 변환합니다."""
        tokens = [self.sos_token] + sentence.split() + [self.eos_token]
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """토큰 ID 리스트를 문장으로 변환합니다."""
        return ' '.join([self.idx2word.get(idx, self.unk_token) for idx in indices])

class TranslationDataset(Dataset):
    def __init__(self, src_data: List[str], trg_data: List[str], 
                 src_vocab: Vocabulary, trg_vocab: Vocabulary, max_len: int):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src = self.src_data[idx]
        trg = self.trg_data[idx]
        
        # 소스 문장을 토큰화하고 패딩
        src_tokens = self.src_vocab.encode(src)
        if len(src_tokens) > self.max_len:
            src_tokens = src_tokens[:self.max_len]
        else:
            src_tokens = src_tokens + [self.src_vocab.pad_idx] * (self.max_len - len(src_tokens))
            
        # 타겟 문장을 토큰화하고 패딩
        trg_tokens = self.trg_vocab.encode(trg)
        if len(trg_tokens) > self.max_len:
            trg_tokens = trg_tokens[:self.max_len]
        else:
            trg_tokens = trg_tokens + [self.trg_vocab.pad_idx] * (self.max_len - len(trg_tokens))
            
        return {
            'src': torch.tensor(src_tokens),
            'trg': torch.tensor(trg_tokens)
        }

def load_data(file_path: str) -> List[str]:
    """파일에서 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """학습과 검증을 위한 데이터로더를 생성합니다."""
    # 데이터 로드
    train_src = load_data(config.train_src)
    train_trg = load_data(config.train_trg)
    valid_src = load_data(config.valid_src)
    valid_trg = load_data(config.valid_trg)
    
    # 어휘 사전 구축
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    src_vocab.build_vocab(train_src, config.vocab_size)
    trg_vocab.build_vocab(train_trg, config.vocab_size)
    
    # 데이터셋 생성
    train_dataset = TranslationDataset(
        train_src, train_trg, src_vocab, trg_vocab, config.max_len
    )
    valid_dataset = TranslationDataset(
        valid_src, valid_trg, src_vocab, trg_vocab, config.max_len
    )
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_dataloader, valid_dataloader, src_vocab, trg_vocab

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 