#-*- coding: utf-8 -*-

"""
(ref) https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/ResNet18_MNIST_Train.ipynb
(ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/03_simple_CNN.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
ResNet18 을 활용해 MNIST 숫자 분류하기 
    * ResNet18 : 학습 가능한 weight가 있는 레이어가 18개 있다는 의미 


1. Create ResNet18 model  

2. Set device 

3. Hyperparameters 

4. Load Data (MNIST)

5. Initialize network 

6. Loss and optimizer 

7. Train network 

8. Check accuracy on training & test to see how good our model
"""

#%% 임포트 토치 
import os.path as osp
import os

import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


# ================================================================= #
#                      1. Create a ResNet18                         #
# ================================================================= #
# %% 01. ResNet18 생성하기 

""" BasicBlock 클래스 정의
ResNet18을 위해 최대한 심플하게 수정 됨 
"""
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        """
        3x3 필터를 사용 (너비와 높이를 줄일 때는 stride 수치 조절)
        """
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        """
        3x3 필터를 사용 (패딩을 1만큼 주기 때문에 너비와 높이가 동일)
        """
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential() # identity인 경우

        """
        # stride가 1이 아니라면, Identity mapping이 아닌 경우
        """
        if stride != 1: 
            self.shortcut = nn.Sequential(  nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes), 
                                        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # (핵심) skip connection
        out = F.relu(out)
        return out



""" ResNet 클래스 정의 
"""
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        """
        64개의 3x3 필터(filter)를 사용
        """
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # gray_level 입력이니 => in_channel = 1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)    # 분류 클래스 10개 => 노드 10개 
        return out


""" ResNet18 함수 정의
"""
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



"""
모델 구조가 잘 만들어 졌는지 확인 

model = ResNet18()      
x = torch.randn(64, 1, 28, 28)    # Batch = 64, 이미지 크기= 28 x 28
print(model(x).shape)   # torch.Size([64, 10])
"""


# ================================================================= #
#                         2. Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')



# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
num_classes = 10 
learning_rate = 0.01
batch_size = 64
num_epochs = 10

load_model = True  # 체크포인트 모델을 가져오려면 True 



# ================================================================= #
#                      4.  Load Data (MNIST)                        #
# ================================================================= #
# %% 04. MNIST 데이터 로드 
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)


transform_train = transforms.Compose([  transforms.ToTensor(), # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81
                                    ])

transform_test = transforms.Compose([   transforms.ToTensor(),
                                    ])



train_dataset = datasets.MNIST( root='dataset/',    # 데이터가 위치할 경로 
                                train=True,         # train 용으로 가져오고 
                                transform=transform_train,  
                                download=True       # 해당 root에 데이터가 없으면 torchvision 으로 다운 받아라 
                                )

train_loader = DataLoader(  dataset=train_dataset,   # 로드 할 데이터 객체 
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True,            # 데이터 순서를 뒤섞어라 
                            num_workers=4,
                            )      


test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform_test, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 


# ================================================================= #
#                      5.  Initialize network                       #
# ================================================================= #
# %% 05. 모델 초기화
model = ResNet18().to(device)
model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = True



# ================================================================= #
#                      9. Checkpoint save & load                    #
# ================================================================= #
#%% 09. 체크포인트를 저장하고 다시 로드하기 

file_name = 'resnet18_mnist.pth.tar'

def save_checkpoint(state, filename=file_name):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



# ================================================================= #
#                  6.  Loss and optimizer  & load checkpoint        #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.SGD( model.parameters(), lr=learning_rate,  momentum=0.9, weight_decay=0.0002)  # 네트워크의 모든 파라미터를 전달한다 

if load_model:
    try: 
        load_checkpoint(torch.load(file_name), model, optimizer)

    except OSError as e: 
        print(e)
        pass 


# ================================================================= #
#                      7.  Train network & save                     #
# ================================================================= #
# %% 07. 학습 루프 

"""
# 학습하기 전에 모델이 AutoGrad를 사용해 학습할 수 있도록 train_mode 로 변환.
(1) backpropagation 계산이 가능한 상태가 됨.
(2) Convolution 또는 Linear 뿐만 아니라, 
    DropOut과 Batch Normalization 등의  파라미터를 가진 Layer들도 학습할 수 있는 상태가 된다. 
"""

"""
for epoch in range(num_epochs):

    model.train()  

    losses = [] 

    
    if epoch % 3 == 0: # 3주기 마다 모델 저장  
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()} # 체크 포인트 상태 
        # Try save checkpoint
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader): # 미니배치 별로 iteration 
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 
        
        
        # forward
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
    
    mean_loss = sum(losses) / len(losses)
    print(f"Loss at epoch {epoch} was {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 
"""



# ================================================================= #
# 8.  Check accuracy on training & test to see how good our model   #
# ================================================================= #
# %% 08. 학습 정확도 확인
"""
(1) 평가 단계에서는 모델에 evaluation_mode 를 설정한다 
    - 학습 가능한 파라미터가 있던 계층들을 잠금 
(2) AutoGrad engine 을 끈다 ; torch.no_grad() 
    - backpropagation 이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임.
    
    - (ref) http://taewan.kim/trans/pytorch/tutorial/blits/02_autograd/
"""

"""
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0

    model.eval()  
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()  #  맞춘 샘플 개수 
            num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')  # 소수 둘 째 자리 까지 표현
"""

# %%
"""
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
"""


# ================================================================= #
#                         Train & Validation 루프                    #
# ================================================================= #

# %% Train 
def train(epoch):
    print(f"\n*****[ Train epoch: {epoch} ]*****")
    model.train()  

    train_losses = [] 
    num_correct = 0 
    num_samples = 0 

    for batch_idx, (data, targets) in enumerate(train_loader): # 미니배치 별로 iteration 
        # Get data to cuda if possible
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 

        # forward
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        train_losses.append(loss.item())

        _, predictions = scores.max(1)
        num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
        num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)

        # backward
        optimizer.zero_grad()   # AutoGrad 하기 전에 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()




        if batch_idx % batch_size == 0:  
            print(f"\n@batch: {str(batch_idx)}")
            print(f"train acc: {str((predictions == targets).sum().item() / predictions.size(0) )}")
            print(f"train loss: {loss.item()}")

    print(f"\nTotal train acc: {float(num_correct)/float(num_samples)*100:.2f}")

    mean_loss = sum(train_losses) / len(train_losses)
    print(f"Mean loss of train: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 

# %% Validation 
def test(epoch):
    print(f"\n*****[ Validation epoch: {epoch} ]*****")
    model.eval()  


    val_losses = [] 
    num_correct = 0 
    num_samples = 0     

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader): # 미니배치 별로 iteration 
            # Get data to cuda if possible
            data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
            targets = targets.to(device=device)  # 레이블 for supervised learning 

            # forward
            scores = model(data)   # 모델이 예측한 수치 
            loss = criterion(scores, targets)
            val_losses.append(loss.item())

            _, predictions = scores.max(1)
            num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
            num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)


        print(f"\nValidation acc: {float(num_correct)/float(num_samples)*100:.2f}")

        mean_loss = sum(val_losses) / len(val_losses)
        print(f"Mean loss of validation: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 


        state = { 'model_state': model.state_dict(), 
                }


        """ 체크 포인트 
        """
        if not osp.isdir('checkpoint_MNIST'):
            os.mkdir('checkpoint_MNIST')

        torch.save(state, osp.join('checkpoint_MNIST',file_name))
        print("=> Model Saved!")




# %% 일정 에폭마다 학습률 줄이기 
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 5:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    
# %% 훈련 루프 실행 
for epoch in range(0, num_epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
# %%
