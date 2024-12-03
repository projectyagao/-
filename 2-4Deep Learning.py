import torch  # PyTorch 라이브러리를 불러옵니다.
import torch.nn as nn  # PyTorch의 신경망 모듈을 불러옵니다.
import torch.optim as optim  # PyTorch의 최적화 알고리즘을 불러옵니다.
import pandas as pd  # 데이터 처리를 위해 Pandas 라이브러리를 불러옵니다.
from torch.utils.data import DataLoader, TensorDataset  # 데이터 로딩과 텐서 데이터를 위한 모듈을 불러옵니다.

# CSV 파일을 읽어서 데이터프레임으로 불러옵니다.
df = pd.read_csv(r'C:\Users\r2com\Desktop\수업자료\data\house_price_of_unit_area.csv')
print(df.head())  # 데이터프레임의 첫 5행을 출력하여 데이터 구조를 확인합니다.

# 'house price of unit area' 열을 라벨 데이터로 분리합니다.
laber_data = df.pop('house price of unit area').values
dataset = df.values  # 나머지 데이터는 입력 데이터셋으로 사용합니다.

# Pytorch 텐서로 변환
dataset = torch.tensor(dataset, dtype=torch.float32)  # 입력 데이터셋을 파이토치 텐서로 변환합니다.
label_data = torch.tensor(laber_data, dtype=torch.float32)  # 라벨 데이터를 파이토치 텐서로 변환합니다.
