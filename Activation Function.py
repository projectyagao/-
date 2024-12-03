#선형 데이터를 비선형 데이터를 액티배이션 함수를 사용
import torch 
import torch.nn as nn # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
import torch.optim as optim # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader # 데이터 관리를 위한 도구들입니다.

class PerceptronModel(nn.module):
    def __init__(self):
        super(PerceptronModel,self).__iinit__()
        self.linear = nn.Linear(1,1)#입력 차원1, 출력 차원 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x