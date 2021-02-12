# ResNet-tutorial
Residual Network 리뷰 및 코딩 연습. <br/>
일반적인 네트워크는 레이거 개수가 일정 이상 증가해버리면 깊어질 수록 성능이 떨어진다. 

* ResNet은 레이어 개수가 깊어진 네트워크를 학습할 수 있게 만든다. 
* 왜냐하면, residual_connection(= shortcut) 부분 때문에 ```H(x)=F(x)``` 를 통으로 학습할 것을 ```H(x) = F(x)+x``` 형태로 만든다. 
* 즉, 이전에 학습된 부분 ```x``` 를 그대로 가져올 수 있어서 다음 스텝에서는 ```F(x)``` 부분만 조금 학습해서 모델을 갱신할 수 있기 때문이다. 


***
### Reference 
[1] [ResNet: Deep Residual Learning for Image Recognition, 꼼꼼한 딥러닝 논문 리뷰와 코드 실습, YouTube](https://youtu.be/671BsKl8d0E) / 논문 리뷰와 코드 실습을 초보자도 이해하기 쉽게 설명해줌. Bottle_neck 구조와 shortcut 설명 주의깊게 듣기.  <br/>
[2] [7.6. Residual Networks, d2l.ai](https://d2l.ai/chapter_convolutional-modern/resnet.html) / 시각화된 자료를 원한다면 참고 <br/>

