A =torch.Tensor([4])
B =torch.Tensor([2])
C =torch.Tensor([1])
D =torch.Tensor([2])
E =torch.Tensor([5])

# 1줄에 torch함수 하나씩만 사용하세요!
out1 = torch.mul(A,B)
out2 = torch.add(C,D)
out3 = torch.sub(out1,out2)

output =torch.sub(out3,E)

# print("result = {}".format(output))

# scalar
s1 = torch.tensor([1.])
s2 =torch.tensor([3.])
add_scalar_12 = s1+s2
# print(s1.size())
# print(s2.size())
# print(add_scalar_12.size())

# 1-dim
v1 =torch.tensor([1.,2.,3.])

v2 =torch.tensor([4.,5.,6.])
add_vector_12 =v1 + v2
# print(v1.size())
# print(v2.size())
# print(add_vector_12.size())

#0부터 9사이의 랜덤 정수3*3크기의 행렬을 만들고, 다른 행렬은 디바이스로 1로 채워진 동일한 크기의 텐서를 생성한 후 두 행렬을 곱해서 결과를 출력해주세여.
# 2-dim
#cpu는 cpu끼리 gpu로할꺼면 gpu로 해야 연산이 가능하다
m1 = torch.randint(0,9, size=(3,3))
m2 =torch.ones_like(m1.cuda())
matrix12 =torch.add(m1.cuda(),m2)
# print(matrix12)
#cuda는 gpu로 연산할때 사용

## data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]])
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]])

# torch.manual_seed: 동일한 결과를 만들 도록 seed를 고정한다.
# torch.rand: [0, 1) 사이의 랜덤 텐서 생성
# torch.randn: 평균=0, 표준편차=1 인 정규분포로부터 랜덤 텐서 생성
# torch.randint: [최저값, 최대값) 사이에서 랜덤 정수 텐서 생성

# 평균 0, 분산 1의 파라미터의 정규분포로 부터 값을 가져옴.
# 학습을 통해 업데이트가 되어 변화되는 모델의 파라미터인 w,b를 의미한다.
#직선의 방정식 y=ax+b

W = torch.randn(1, 1) #기울기

b = torch.randn(1, 1) #편차

# print("W : ", W)
# print("b : ", b)

for j in range(len(x_data)): #4개만큼 for문이 돈다(x_data갯수만큼  : 4번)
    # data * weight 작성 
    #WX = x_data[j] * W # perceptron 모델
    WX = torch.matmul(x_data[j],W) #matmul=곱하기

    ## bias add 작성
    y_hat = torch.add(WX,b) #add = 더하기

    ## W와 b로 예측 하기 y_dat = y의실제값   prediction = y의 예측값
    print("y_data: , ",y_data[j], "prediction : ", y_hat)


import torch
## data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]])
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]])

# 평균 0, 분산 1의 파라미터의 정규분포로 부터 값을 가져옴.
# 학습을 통해 업데이트가 되어 변화되는 모델의 파라미터인 w,b를 의미한다.

W = torch.nn.Parameter(torch.normal(mean=0,std=1,size=(1,1)))
b =torch.nn.Parameter(torch.normal(mean=0,std=1,size=(1,1)))
lr = torch.tensor(0.0001)
print("W : ", W)
print("b : ", b)
print('lr : ', lr)

for i in range(2000):  ## 에폭
    total_error = 0

    for j in range(len(x_data)): ## 배치 1
        ## data * weight
        WX = torch.matmul(x_data[j],W)
        # (1, 1) * (1, 1)

        ## bias add
        y_hat = torch.add(WX,b)

        ## 정답인 Y와 출력값의 error 계산
        error = torch.subtract(y_data[j],y_hat) ##(true - prediction)

        ## 경사하강법으로 W와 b 업데이트.
        ## 도함수 구하기
        diff_W = torch.matmul(error,x_data[j]) #error*x
        diff_b =error

        ##  업데이트할 만큼 러닝레이트 곱
        diff_W = torch.multiply(lr,diff_W) 
        diff_b = torch.multiply(lr,diff_b) #lr * (error)

        ## w, b 업데이트
        W = torch.add(W,diff_W) #w + lr * (error)
        b =  torch.add(b,diff_b) #b + lr * (error)

        ## 토탈 에러.
        visual_error = torch.square(error)
        total_error = total_error + visual_error


    ## 모든 데이터에 따른 error 값
    print("epoch: ", i, "error : ", total_error/len(x_data))
    


#inference
torch.add(torch.multiply(10,W),b)