### 👽[Kaggle]E.T. Signal Search

##### 🚩 Find extraterrestrial signals in data from deep space
##### 📆 2021.05.20~2021.08.01
---
![ET](https://user-images.githubusercontent.com/65913073/120472505-ef4fb100-c3e0-11eb-9155-b412f43488a9.png)

| *"Are we alone in the Universe?"*

### 📜 대회 정보 
- Spectogram data를 이용하여 signal을 분류하는 classification task 입니다

| *👽Target image1* | *👽Target image2* | *👽Taget image3* | 
| ------------ | ------------ | ------------ |
|![S.G](https://user-images.githubusercontent.com/65913073/120474040-a7318e00-c3e2-11eb-9651-2becf08af7a8.png)|![S.G](https://user-images.githubusercontent.com/65913073/120474045-a862bb00-c3e2-11eb-889e-4d06520a5fc1.png)|![S.G](https://user-images.githubusercontent.com/65913073/120474049-a993e800-c3e2-11eb-8ba4-f3492e89cd11.png)

### 💡 문제 정의 및 해결 방안
- 현재 진행중인 대회로 parameter 및 시도한 방법들은 [link](https://vimhjk.oopy.io/3a624cba-ecb1-4d4c-bebc-dca5493b6198)에 정리되어 있습니다.

### 📑 *code*
```
______config
|      |____config.yml ### train config
|      |____eval_config.yml  ### evaluation config
|
|______src
|      |____dataset.py ### dataset
|      |____model.py  ### cnn model
|      |____train.py  ### train, validation module
|      |____utils.py   ### utils
|
|
|____main.py  ### run
|
|____eval.py ### evaluation


```

### 💡 *Getting Start*

### train
`python train.py --config your_config_name`
### evaluate
`python inference.py --config your_config_name`

