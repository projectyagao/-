import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 선언
x_data = torch.tensor([[5.], [30.], [95.], [100.], [265.], [270.], [290.], [300.], [365.]], dtype=torch.float32)
y_data = torch.tensor([[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]], dtype=torch.float32)



# 퍼셉트론 모델 구현
class PerceptronModel(nn.Module):
    def __init__(self):
        super(PerceptronModel, self).__init__()
        #
        #

    def forward(self, x):
        x = #
        x = #
        return x