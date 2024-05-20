import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.data import DataLoader
import torchvision


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # character dictionary
        self.index_to_char = {i:ch for i, ch in enumerate(sorted(list(set(text))))}
        self.char_to_index = {ch:i for i, ch in self.index_to_char.items()}
        self.indexed_text = [self.char_to_index[char] for char in text]
        self.seq_len = 30

        print(len(self.char_to_index))


    def __len__(self):
        return len(self.indexed_text) - self.seq_len

    def __getitem__(self, idx):
        input = torch.tensor(self.indexed_text[idx : idx+self.seq_len])
        target = torch.tensor(self.indexed_text[idx+1 : idx+self.seq_len+1])

        return input, target

if __name__ == '__main__':
    dataset = Shakespeare('./shakespeare_train.txt')
    input, target = dataset[0]
    # 첫 번째 데이터 샘플 출력
    print(input)  
    print(target)