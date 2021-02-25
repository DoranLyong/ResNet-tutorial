#-*- coding: utf-8 -*-

#%%
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 

#%% residual-block 정의 
class block(nn.Module):
    def __init__(
        self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__() # 상속받은 부모 클래스의 모든 attributes을 그대로 받아옴 
        self.expansion = 4

        self.conv1 = nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0 )
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv2 = nn.Conv2d( out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d( out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample


    def forward(self, x):
        identity = x.clone() # identity_value 저장

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


#%%
class ResNet(nn.Module):
    """ layers := residual_block 의 개수를 리스트 형태로 담고 있음 
    (e.g) layers = [3, 4, 6, 3]
    """
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__() # 상속받은 부모 클래스의 모든 attributes을 그대로 받아옴 
        
        
        """ conv1 정의 
        """
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        """ ResNet layers 정의 
        나머지 conv2_x, conv3_x, conv4_x, conv5_x  정의 
        """
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer( block, layers[0], out_channels=64, stride=1 )
        self.layer2 = self._make_layer( block, layers[1], out_channels=128, stride=2 )
        self.layer3 = self._make_layer( block, layers[2], out_channels=256, stride=2 )
        self.layer4 = self._make_layer( block, layers[3], out_channels=512, stride=2 )

        """ 마지막 레이어 
        """
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if (stride != 1) or (self.in_channels != out_channels * 4):
            identity_downsample = nn.Sequential(    nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                                                    nn.BatchNorm2d(out_channels * 4),
                                                )


        """ Residual_block 추가 
        """
        layers.append( block(self.in_channels, out_channels, identity_downsample, stride) )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = out_channels * 4

        """ For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
            then finally back to 256. Hence no identity downsample is needed, since stride = 1,
            and also same amount of channels.
        """
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


#%% 모델 타입 
def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)




#%% 테스트 
if __name__ == "__main__":
    # 프로세스 장비 설정 
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")


    # 모델 초기화 
    model = ResNet152(img_channel=3, num_classes=1000).to(device)
    print(model)

    # 모델 출력 테스트 
    x = torch.randn(4, 3, 224, 224).to(device)
    print(model(x).shape)
# %%
