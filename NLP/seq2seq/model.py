import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """시퀀스-투-시퀀스 모델의 인코더 부분을 구현한 클래스입니다.

    이 클래스는 입력 시퀀스를 받아 임베딩하고 LSTM을 통해 인코딩합니다.
    인코딩된 결과는 디코더에서 사용할 수 있는 형태로 반환됩니다.

    Args:
        vocab_size (int): 입력 어휘의 크기
        hidden_size (int): LSTM의 히든 상태 크기
        num_layers (int): LSTM 레이어의 수
        dropout (float): 드롭아웃 비율 (num_layers > 1일 때만 적용)
    """
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # 입력 텐서의 차원 순서:
                             # True:  [batch_size, sequence_length, feature_size]
                             # False: [sequence_length, batch_size, feature_size] (기본값)
        )
        
    def forward(self, x):
        """입력 시퀀스를 인코딩합니다.

        Args:
            x (torch.Tensor): 입력 시퀀스
                Shape: [batch_size, seq_len]

        Returns:
            tuple: (outputs, (hidden, cell))
                - outputs (torch.Tensor): 각 타임스텝의 LSTM 출력
                    Shape: [batch_size, seq_len, hidden_size]
                - hidden (torch.Tensor): 마지막 타임스텝의 히든 상태
                    Shape: [num_layers, batch_size, hidden_size]
                - cell (torch.Tensor): 마지막 타임스텝의 셀 상태
                    Shape: [num_layers, batch_size, hidden_size]
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    """시퀀스-투-시퀀스 모델의 디코더 부분을 구현한 클래스입니다.

    이 클래스는 인코더의 출력을 받아 타겟 시퀀스를 생성합니다.
    LSTM을 사용하여 시퀀스를 생성하고, 각 타임스텝마다 다음 토큰을 예측합니다.

    Args:
        vocab_size (int): 출력 어휘의 크기
        hidden_size (int): LSTM의 히든 상태 크기
        num_layers (int): LSTM 레이어의 수
        dropout (float): 드롭아웃 비율 (num_layers > 1일 때만 적용)
    """
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell):
        """디코더의 한 타임스텝을 처리합니다.

        Args:
            x (torch.Tensor): 현재 타임스텝의 입력 토큰
                Shape: [batch_size, 1]
            hidden (torch.Tensor): 이전 타임스텝의 히든 상태
                Shape: [num_layers, batch_size, hidden_size]
            cell (torch.Tensor): 이전 타임스텝의 셀 상태
                Shape: [num_layers, batch_size, hidden_size]

        Returns:
            tuple: (prediction, hidden, cell)
                - prediction (torch.Tensor): 다음 토큰에 대한 예측 확률
                    Shape: [batch_size, 1, vocab_size]
                - hidden (torch.Tensor): 현재 타임스텝의 히든 상태
                    Shape: [num_layers, batch_size, hidden_size]
                - cell (torch.Tensor): 현재 타임스텝의 셀 상태
                    Shape: [num_layers, batch_size, hidden_size]
        """
        embedded = self.embedding(x)  # [batch_size, 1, hidden_size]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)  # [batch_size, 1, vocab_size]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """시퀀스-투-시퀀스 모델의 전체 구조를 구현한 클래스입니다.

    이 클래스는 인코더와 디코더를 결합하여 시퀀스 변환 작업을 수행합니다.
    인코더는 입력 시퀀스를 인코딩하고, 디코더는 인코딩된 정보를 바탕으로
    타겟 시퀀스를 생성합니다. Teacher forcing을 사용하여 학습의 안정성을 높입니다.

    Args:
        encoder (Encoder): 입력 시퀀스를 인코딩하는 인코더 모델
        decoder (Decoder): 타겟 시퀀스를 생성하는 디코더 모델
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """시퀀스-투-시퀀스 모델의 순전파를 수행합니다.

        Args:
            src (torch.Tensor): 소스 시퀀스
                Shape: [batch_size, src_len]
            trg (torch.Tensor): 타겟 시퀀스
                Shape: [batch_size, trg_len]
            teacher_forcing_ratio (float, optional): Teacher forcing을 사용할 확률
                기본값: 0.5

        Returns:
            torch.Tensor: 각 타임스텝에서의 예측값
                Shape: [batch_size, trg_len, vocab_size]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        # 출력을 저장할 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        # 인코더의 마지막 hidden state를 디코더의 첫 hidden state로 사용
        _, (hidden, cell) = self.encoder(src)
        
        # 첫 번째 입력은 <sos> 토큰
        input = trg[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing 사용 여부 결정
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs 