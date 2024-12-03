import torch 
import torch.nn as nn # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
import torch.optim as optim # 신경망 및 최적화 도구를 사용하기 위해 임포트합니다.
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader # 데이터 관리를 위한 도구들입니다.

x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.],[1.,1.]], dtype=torch.float32)
y_data = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

#퍼셉트론 모델 구현
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron,self).__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,5)
        self.output_layer = nn.Linear(5,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
    
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
    
#모델 인스턴스 생성
model = MultiLayerPerceptron()

#손실 함수 및 옵티마이저 정의
optimizer = optim.SGD(model.parameters(),lr=0.001)
criterion = nn.BCELoss()

#모델 학습
epochs = 2000
for epoch in range(epochs):
    #Forward pass
    outputs = model(x_data)
    loss = criterion(outputs,y_data)
    
    #Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (torch.argmax(outputs,dim=1) == y_data).float().sum()
    
   # if (epoch+1) % 200 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터 준비
test_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], dtype=torch.float32)

# 모델을 통해 테스트 데이터 예측
with torch.no_grad():  # Gradient 계산을 수행하지 않음
    predictions = model(test_data)
    print("Test Data 예측 값:")
    for i, test_val in enumerate(test_data, start=1):
        print(f" test data {test_val.numpy()} 예측 값 : {(predictions>0.5).float()}")    