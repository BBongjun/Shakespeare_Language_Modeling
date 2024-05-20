# Shakespeare Language Model
Text generation model(CharRNN, CharLSTM)

## Setups
- numpy : 1.26.0
- Python : 3.9.18
- pytorch : 2.1.1

## Experiment Setting
- epoch : 100
- batch_size : 128
- optimizer : AdamW (to prevent overfitting)
- lr : 0.001 
- embedding dim : 128
- RNN, LSTM num_layers : 2
- hidden_size : 512

## File description
- main.py : for train(+ option: text generation)
- generate.py : for text generation
- model.py : model definition
- dataset.py : Shakespeare dataset & dataloader

## Run

```
python main.py
```

## The average loss values for training and validation
- RNN
![RNN_avg_loss_plot]()

- LSTM
![LSTM_avg_loss_plot]() 

- RNN vs LSTM 
![RNN_vs_LSTM]()

####
- RNN val loss > LSTM val loss
- RNN보다 LSTM이 더 자연스러운 텍스트를 생성할 것으로 기대됨.

## Text Generation Performance
- RNN과 LSTM의 텍스트 생성 능력 확인
    1. Tuning a temperature(T) in Softmax function
    2. RNN vs LSTM performance


### 1. Tuning a temperature(T) in Softmax function
#### RNN text generation
```

```
