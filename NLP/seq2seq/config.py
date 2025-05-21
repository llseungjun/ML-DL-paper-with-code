import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Seq2Seq 모델 학습을 위한 설정')
    
    # 데이터 관련 설정
    parser.add_argument('--vocab_size', type=int, default=30000,
                      help='어휘 사전의 크기')
    parser.add_argument('--max_len', type=int, default=50,
                      help='최대 시퀀스 길이')
    
    # 모델 관련 설정
    parser.add_argument('--hidden_size', type=int, default=512,
                      help='LSTM의 히든 상태 크기')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='LSTM 레이어의 수')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='드롭아웃 비율')
    
    # 학습 관련 설정
    parser.add_argument('--batch_size', type=int, default=64,
                      help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='학습률')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='학습 에폭 수')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='학습에 사용할 디바이스 (cuda 또는 cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='랜덤 시드')
    
    # 데이터 경로 설정
    parser.add_argument('--train_src', type=str, required=True,
                      help='학습 데이터 소스 파일 경로')
    parser.add_argument('--train_trg', type=str, required=True,
                      help='학습 데이터 타겟 파일 경로')
    parser.add_argument('--valid_src', type=str, required=True,
                      help='검증 데이터 소스 파일 경로')
    parser.add_argument('--valid_trg', type=str, required=True,
                      help='검증 데이터 타겟 파일 경로')
    
    # 저장 경로 설정
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='모델 체크포인트 저장 디렉토리')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='TensorBoard 로그 저장 디렉토리')
    
    args = parser.parse_args()
    return args

class Config:
    def __init__(self, args=None):
        if args is None:
            args = get_args()
            
        # 데이터 관련 설정
        self.vocab_size = args.vocab_size
        self.max_len = args.max_len
        
        # 모델 관련 설정
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        
        # 학습 관련 설정
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        
        # 기타 설정
        self.device = args.device
        self.seed = args.seed
        
        # 데이터 경로
        self.train_src = args.train_src
        self.train_trg = args.train_trg
        self.valid_src = args.valid_src
        self.valid_trg = args.valid_trg
        
        # 저장 경로
        self.save_dir = args.save_dir
        self.log_dir = args.log_dir 