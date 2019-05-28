## README

다음의 순서로 실행이 된다.
+ 특이하게도 keras를 사용함에도 불구하고 ch-first 방식으로 코딩을 하였다.
keras model의 data format은 ch-last가 default지만 ch-first방식으로 설정이 가능하다.
구현한 model을 보면 ch-first 방식으로 입력을 받게 만들어놔 큰 문제 없이 돌아갈것으로 예상된다.
하지만 굳이 ch-first로 해야 할 필요가 있었을까 하는 생각이 든다.

### Data 준비 단계
#### 1. prepare_datasets_DRIVE.py

이는 최초 DRIVE data를 준비하기위한 파일이다.
따라서 최초 한번만 실행을 해주면 된다. 

최초 실행시 해당 파일의 PATH부분을 수정해주어야 한다.


### Training Session
#### 1. 어떻게 Patch를 생성해 내는가?
먼저 우리는 patch의 크기, 생성해 낼 patch 수를 정한 후, 이에 상응하는 patch를 생성해낸다.
그렇다면 이러한 random patch는 어떻게 생성이 되는 것 일까?
- **get_data_training** (in ./lib/extract_patches.py)
해당 함수를 통해서 patch가 생산이 된다.
본 함수에는 extract_random 라는 함수가 call 되어 정해진 수 만큼의 patch를 생성해낸다.
함수의 흐름은 다음과 같다. Data load -> Preprocessing(이때 ch이 1이 됨) -> Data resize (565  * 565) -> extract_random
- extract_random
각 이미지마다 patch를 생성하고 생성한 patch를 return 하는 함수이다.
함수의 흐름은 다음과 같다.
ch-first형태의 patch를 담은 frame 형성(num_patch, ch, h, w) -> 한 이미지당 생성할 패치를 정한다 (전체 패치수 / 원본 이미지의 수)
-> Loop(원본 이미지의 수만큼) 내부 Loop (patch_per_img수 만큼 반복), 앞에서 정한 패치크기만큼의 각 이미지마다 random patch 생성

해당 함수를 거치고 난 후, model.fit을 통하여 patch를 학습시킨다.
따라서 내가 Generator를 만들때 training 부분만 잘 건드려주면 이를 성공적으로 만들 수 있을거라 예상된다.

### Test Session
validation dataset, patch의 크기, stride 크기를 입력받는다. stride는 average mode (overlap-tile 일 경우에 유효하다.)
해당 Session 의 흐름은 다음과 같다.
이를 바탕으로 validation set의 patch를 뽑는다. -> model이 patch를 예측 -> 잘개 쪼개진 patch를 recompose 시켜준다. -> 성능 평가

#### Non overlap-tile 일 경우
##### 1. Patch 생성단계
- **get_data_testing**
본 함수는 validation dataset의 patch를 생성해내는 함수이다.
큰 흐름은 다음과 같다.
validation dataset load -> preprocessing -> paint border (zero padding & new img frame) -> extract_ordered
- paint order
이미지 원본과 최초 설정한 patch의 크기가 parameter로 들어간다.
원본 이미지를 patch 사이즈로 나눌때, 정수로 나누어 떨어지지 않는 경우가 매우 많을 것이다.
new_img_h = ((int(img_h/patch_h)) +1) * patch_h
new_img_w = ((int(img_w/patch_w)) +1) * patch_w
라는 코드를 볼 수 있는데 이는 patch size로 나누어 떨어지는 새로운 image frame을 생성해내기 위한 코드이다.
**zero-padding** 을 하여 새로운 frame을 생성하게 된다.
또한 data의 shape 도 바꾸어 준다.
원본 데이터의 shape는 (20, 3, 584, 565) 이지만 함수를 거치고 나온 데이터의 shape는 (20, 584, 565, 3)이다. 
이는 데이터를 Tensor형으로 바꾸어 준 것인데 이미지 tensor는 (batch size, h,w,c) or (batch size, c,h,w) 로 표현이 되는데,
Tensorflow에서는 ch-last 방식(전자)을 사용하고 theano에서는 ch-first방식(후자)를 주로 사용한다. 여기서는 ch-last로 바꾸어주었다.
- extract_ordered
이미지, patch의 크기를 파라미터로 받는다.
큰 흐름은 다음과 같다.
각 이미지마다 뽑을 patch수 및 전체 patch를 담을 frame 생성 -> 순차적으로 image로부터 patch를 생성한다.
즉, training과 달리, random 하지 않게 patch를 생성하여 예측을 시킨다. (random 할 필요가 없다.)

위로부터 얻어진 이미지의 patch를 model을 통하여 예측한다.

##### 2. Patch를 붙히는 단계 (Recompose)
- **recompose_img** 
patch로 쪼개진 이미지를 복원하는 단계이다.
이미지, (img_h / patch_h), (img_w / patch_w) 로 얻어진 이미지의 높이와 너비당 생성한 패치의 갯수를 파라미터로 넣어준다. 
함수의 흐름은 다음과 같다.
복원된 single 이미지들을 담을 data frame을 생성한다. -> 각 이미지들을 높이와 너비 마다 생성한 패치의 갯수만큼 loop를 돌려 하나씩 붙힌다. -> return


#### Overlap-tile 일 경우
overlap tile은 약간씩 겹치는 이미지를 생성하고, 각 이미지의 겹치는 부분을 더하여 평균값을 취하는 일종의 앙상블과 같은 방법이다.
많은 이미지들의 평균으로 얻어지므로, 좀더 신뢰 할 수 있는 좋은 결과를 얻게 된다.
전체적인 흐름은 non-overlap과 같다. 

- **get_data_testing_overlap**
위와 유사하지만, Stride 정보가 추가적으로 파라미터로써 들어온다.
전체적인 흐름은 
validation dataset load -> preprocessing -> paint border overlap(zero padding & new img frame) -> extract_ordered_overlap
따라서 위의 overlap함수들을 잘 살펴보는게 중요하다. 
- paint_border_overlap
결론적으로 말하면 padding을 해주는 단계이다.
stride가 추가적으로 들어온다.
다음과 같은 부분이 추가되었음을 확인 할 수있다.
leftover_h = (img_h - patch_h) % stride_h
leftover_w = (img_w - patch_w) % stride_w
이는 일반적인 padding 공식으로부터 차용한것인데, patch사이즈만큼의 filter를 가지고 stride를 할 것인데,
padding을 해야하는지? 를 묻고있는 것이다.
만약 0이아니라면, data frame을 만들어 필요한 공간만큼 padding을 하고, 기존 이미지를 넣어준다.
- extract_ordered_overlap
non-overlap 보다 당연히 많은 양의 patch를 뽑아낸다.
일부 겹치는 부분까지도 뽑아내겠다고 허용해주었기 때문이다.
**이전에는 patch size만큼의 stride를 해주었다고 생각하면 직관적으로 이해가 딱 된다.**
최종적으로 다음의 숫자를 가지는 patch를 뽑아낸다.
num_patch_img = ((img_h - patch_h)//stride_h +1) * ((img_w - patch_w)//stride_w +1)
그 후, 각 이미지마다 stride를 허용하여 patch를 뽑아내고 저장하고 return한다.
non-tile 일때 patch size만큼의 stride를 했다는것을 이해하는 순간 이 부분은 바로 이해가 갈듯...!

- **recompose_img_overlap**
똑같다. 하지만 stride를 고려해준다는것만 잊지않으면 된다.
그리고 한 이미지로 부터 얻어진 모든 patch를 더한후 평균을 내준다.
그게 전부다.

#### FOV에 관하여
FOV를 보는 함수가 존재한다. pred_only_FOV, inside_FOV_DRIVE
DRIVE 는 동그라미 형태의 FOV를 가지는데,
해당 영역내에서만 보겠다는 의미이다. 이는 inside_FOV_DRIVE로 판별한다.


