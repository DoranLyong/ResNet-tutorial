# ResNet-tutorial
Residual Network 리뷰 및 코딩 연습. <br/><br/>

일반적으로 평평하게 쌓는 네트워크(=plain network)는 레이어 개수가 깊어질수록 학습 난위도가 커지면서 성능은 떨어진다. <br/>
왜냐하면, 레이어가 증가하는 만큼 학습해야할 파라미터 개수도 증가하기 때문에 수렴 난위도가 높아진다 (e.g., vanishing/exploding gradients, degradation problem). 

* ResNet은 레이어 개수가 깊어진 네트워크를 학습할 수 있게 만든다. 
* 왜냐하면, skip_connection(= shortcut) 부분 때문에 ```H(x)=F(x)``` 를 통으로 학습할 것을 ```H(x) = F(x)+x``` 형태로 만든다. 
* 즉, 앞서 학습된 정보 ```x```(=identity map) 를 그대로 가져올 수 있어서 잔여 부분인 ```F(x)```만 조금만 학습해서 모델을 갱신할 수 있기 때문이다 (즉, 전체를 한번에 학습하는 것 보다 ```조금씩 점진적으로 학습하니까``` 쉬워진다). 
* 덕분에 레이어가 더 깊어질수록 성능 또한 증가한다. 


※ [dimension matching](https://youtu.be/671BsKl8d0E?t=1628) 부분이 어떻게 구현됐는지 보기 <br/>
※ [Identity vs. Projection Shortcuts](https://youtu.be/671BsKl8d0E?t=2032) / shorcut 부분도 학습 가능하게 만들까? 말까? 결론은 굳이 안 해도 됨. <br/>
※ [Bottleneck architectures](https://youtu.be/671BsKl8d0E?t=2079) / Identity shortcut을 사용하면 bottleneck 구조의 복잡도를 줄여준다. 


***
### Reference 
[1] [ResNet: Deep Residual Learning for Image Recognition, 꼼꼼한 딥러닝 논문 리뷰와 코드 실습, YouTube](https://youtu.be/671BsKl8d0E) / 논문 리뷰와 코드 실습을 초보자도 이해하기 쉽게 설명해줌. Bottle_neck 구조와 shortcut 설명 주의깊게 듣기.  <br/>
[2] [7.6. Residual Networks, d2l.ai](https://d2l.ai/chapter_convolutional-modern/resnet.html) / 시각화된 자료를 원한다면 참고 <br/>
[3] [Deep Residual Learning for Image Recognition, CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) / 논문 링크 <br/>
[4] [paper with code](https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition) / 코드 링크 

