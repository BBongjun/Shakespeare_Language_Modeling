# import some packages you need here
import torch
import dataset
from model import CharRNN, CharLSTM
from torch.utils.data import DataLoader
import argparse

def save_generated_text(filename, texts):
    with open(filename, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + '\n\n')

def generate(model, seed_characters, temperature, device, dataset, len = 150, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    input = torch.tensor([dataset.char_to_index[s] for s in seed_characters]).unsqueeze(0).to(device)

    hidden = model.init_hidden(1)
    # 숨겨진 상태와 셀 상태를 각각 디바이스로 이동
    if isinstance(hidden, tuple):  # LSTM의 경우
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:  # RNN의 경우
        hidden = hidden.to(device)

    samples = []
    with torch.no_grad():
        for i in range(len):
            output, hidden = model(input, hidden)
            distribution = torch.softmax(output[-1].squeeze() / temperature, dim=0)
            #predicted char_index by multinomial
            char_index = torch.multinomial(distribution, num_samples=1).item()
            #input(index) update
            input = torch.cat((input, torch.tensor([[char_index]]).to(device)), dim=1)
            # 예외 처리 추가
            samples.append(dataset.index_to_char[char_index])
    samples = ''.join(samples)

    return samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shakespeare_LM')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--generate', default=True, type=bool)
    parser.add_argument('--gen_len', default=300, type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Construct dataloader / (size) train:val = 0.8:0.2
    full_dataset = dataset.Shakespeare('./dataset/shakespeare_train.txt')

    # Model init
    vocab_size = len(full_dataset.index_to_char)
    #print(vocab_size)
    model_rnn = CharRNN(vocab_size, args.embedding_dim, args.hidden_size, vocab_size, args.n_layers).to(device)
    model_lstm = CharLSTM(vocab_size, args.embedding_dim, args.hidden_size, vocab_size, args.n_layers).to(device)

    model_rnn.load_state_dict(torch.load(f'./save_model/best_model_rnn_epoch93_layers_2.pth'))
    best_rnn = model_rnn
    model_lstm.load_state_dict(torch.load(f'./save_model/best_model_lstm_epoch97_layers_2.pth'))
    best_lstm = model_lstm

    seed_characters_list = ['QUEEN: So, lets end this', 'Lord: Kill him!', 'Citizen: I have a bad news.', 'Princess: I love you', 'All: Wow']
    temperatures = [0.1, 0.3, 0.5, 0.8, 1, 1.5, 2]

    rnn_texts = []
    lstm_texts = []
    for seed_characters in seed_characters_list:
        for temperature in temperatures:
            print(f'================= temperature : {temperature} =================')
            
            rnn_gen_text = generate(best_rnn, seed_characters, temperature, device, full_dataset, len = args.gen_len)
            lstm_gen_text = generate(best_lstm, seed_characters, temperature, device, full_dataset, len = args.gen_len)
            rnn_texts.append(f"Temperature {temperature}, Seed: {seed_characters}\n{rnn_gen_text}")
            lstm_texts.append(f"Temperature {temperature}, Seed: {seed_characters}\n{lstm_gen_text}")

            print(f'################ Temperature {temperature}, Seed: {seed_characters} ################')
            print('==========Generated text by Vanilla RNN==========')
            print(rnn_gen_text)
            print('==========Generated text by LSTM==========')
            print(lstm_gen_text)

    save_generated_text('generated_text/generated_texts_rnn.txt', rnn_texts)
    save_generated_text('generated_text/generated_texts_lstm.txt', lstm_texts)
