# deep_learning_models_implementation
논문이나 블로그를 참고하여 딥러닝 모델을 직접 구현하고 정리하기 위한 repo 입니다.



## 0. Tensorflow custom trainer 구현
- 참고 공식 자료: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=ko
- checkpoint 저장, best metric checkpoint 저장, tensorboard를 위한 logging이 포함되어 있습니다.

## 1. ResNet implementation in Tensorflow (ResNet34, ResNet50)
- 참고 논문: https://doi.org/10.48550/arXiv.1512.03385
- 논문 이해하기(blog): https://heesunpark26.tistory.com/17
- 논문 구현하기(blog): https://heesunpark26.tistory.com/20
- 참고 사항: Conv2D 이용, 2D input 가정

## 2. MobileNet implementation in Tensorflow
- 참고 논문: https://doi.org/10.48550/arXiv.1704.04861
- 논문 이해하기(blog): https://heesunpark26.tistory.com/26
- 참고 사항: Conv1D 이용, 1D input 가정

## 3. Transformer implementation in Pytorch
- 참고한 블로그: https://cpm0722.github.io/pytorch-implementation/transformer
