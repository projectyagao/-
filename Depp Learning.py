import torch 
import torch.nn as nn # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
import torch.optim as optim # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader # 데이터 관리를 위한 도구들입니다.


# data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]]) #입력 데이터, 4개의 데이터 포인트로 구성된 1차원 배열
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]]) #타겟 데이터, 입력 데이터에 대한 정답

#----------------------외우기--------------------------------------------------------
# 1-layer perceptron 만들기
linear = torch.nn.Linear(1,1, bias=True) # 간단한 선형 변환 (y = Wx + b)을 정의합니다. 입력과 출력이 1차원이고, bias(절편)을 사용합니다.
model = torch.nn.Sequential(linear) # 모델을 순차적으로 실행할 수 있도록 nn.Sequential로 묶어줌

# 학습에 사용할 데이터를 TensorDataset으로 묶어줍니다. 
# x_data: 입력 데이터, y_data: 정답(레이블) 데이터입니다.
dataset = TensorDataset(x_data,y_data)

# 데이터를 작은 단위로 나누어 학습할 수 있도록 DataLoader를 만듭니다.
# batch_size: 한 번에 학습할 데이터 개수, shuffle: 데이터 순서를 무작위로 섞을지 여부입니다.
batch_size = 1
data_loader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)

# 선형 회귀 모델을 직접 정의하는 클래스입니다.
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__() # nn.Linear를 사용해 선형 회귀 계층을 정의합니다.
        self.linear = nn.Linear(1,1) # 입력 1개, 출력 1개인 선형 계층
    def forward(self, x): 
        # 모델의 예측을 수행하는 함수입니다.
        # 주어진 x값에 대해 y = Wx + b 계산을 실행합니다.
        return self.linear(x)

# 위에서 정의한 LinearRegression 모델을 생성합니다.
model = LinearRegression()

# 손실 함수 정의: MSE(평균 제곱 오차)입니다.
# 예측값(y_pred)과 실제값(y_data) 간의 차이를 계산합니다.
criterion = nn.MSELoss()

# 최적화 알고리즘 정의: SGD(확률적 경사 하강법)입니다.
# 학습률(lr)은 0.01로 설정합니다.
optimizer=optim.SGD(model.parameters(),lr=0.01)

# 학습 과정 (총 2000번 반복)
epochs = 2000 # 학습을 몇 번 반복할지 설정
for epoch in range(epochs):
    model.train()  # 모델을 학습 모드로 전환
    
    for x_batch, y_batch in data_loader:   # 배치 단위로 데이터를 가져옵니다.
        optimizer.zero_grad()  # 이전 배치에서 계산된 기울기를 초기화합니다.
        y_pred = model(x_batch)  # 모델을 사용하여 예측값을 계산합니다.
        loss = criterion(y_pred, y_batch)  # 예측값과 실제값 간의 손실(오차)을 계산합니다.
        loss.backward()  # 손실을 기준으로 기울기를 계산합니다.
        optimizer.step()  # 계산된 기울기를 사용해 모델의 가중치와 절편을 업데이트합니다.

      # 100번마다 학습 진행 상황을 출력합니다.
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}") # 현재 손실값 출력

# 학습 완료 후 모델의 가중치와 절편을 출력합니다.
for name, param in model.named_parameters():
    if param.requires_grad: # 학습 가능한 파라미터만 출력합니다.
        print(name, param.data) # 파라미터 이름과 값을 출력
#----------------------------------------- 외우기----------------------------------------
        
        
# model.eval() #모델을 평가 모드로 전환 이는 예측시 사용
# with torch.no_grad(): #평가 모드에서는 기울기를 걔산하지 않기 위해 WITH torch.no_grad()블록을 사용
#     predict = model(x_data) #입력 데이터에 대한 예측을 수행
#     predict = predict.cpu().data.numpy() #예측 결과를 cpu로 이동시키고  numpy 배열로 변환
#     print('train:', x_data) #입력 데이터를 출력
#     print('predict:', predict) #예측 결과를 출력



