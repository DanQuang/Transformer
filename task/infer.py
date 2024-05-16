import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model.transformer import Transformer
from data_utils.load_data import Load_Data
import pandas as pd


class Infer_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.dropout = config["dropout"]
        self.batch_size = config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load text embedding config
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.seq_len = config["text_embedding"]["max_length"]
        
        # Load model config
        self.n_layers = config["model"]["n_layers"]
        self.n_head = config["model"]["n_head"]
        self.hidden = config["model"]["hidden"]

        # Load data
        self.data_path = config["data_path"]
        self.load_data = Load_Data(self.data_path, self.seq_len, self.batch_size)

        # Load vocab
        self.vocab_en = self.load_data.vocab_en
        self.vocab_de = self.load_data.vocab_de

        # Load vocab size and pad idx src and trg
        self.en_vocab_size = self.vocab_en.vocab_size()
        self.de_vocab_size = self.vocab_de.vocab_size()
        self.en_pad_idx = self.vocab_en.pad_idx()
        self.de_pad_idx = self.vocab_de.pad_idx()

        # Load model
        self.model = Transformer(src_pad_idx= self.de_pad_idx,
                                 trg_pad_idx= self.en_pad_idx,
                                 enc_vocab_size= self.de_vocab_size,
                                 dec_vocab_size= self.en_vocab_size,
                                 embedding_dim= self.embedding_dim,
                                 n_head= self.n_head,
                                 seq_len= self.seq_len,
                                 hidden= self.hidden,
                                 n_layers= self.n_layers,
                                 dropout= self.dropout).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index= self.en_pad_idx)

    def predict(self):
        test = self.load_data.load_test()

        best_model = "Transformer_best_model.pth"

        if os.path.join(self.save_path, best_model):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])

            
            self.model.eval()
            with torch.inference_mode():
                epoch_loss = 0
                predict_tokens_list = []
                gt = [data["en_sentence"] for data in self.load_data.test_dataset]
                for _, item in enumerate(tqdm(test)):
                    source, target = item["de_ids"].to(self.device), item["en_ids"].to(self.device)
                    output = self.model(source, target)

                    predict_token = output.argmax(-1)
                    predict_tokens_list.append(predict_token)

                    # output: [batch_size, target_len, target_vocab_size]
                    output_dim = output.shape[-1] # target_vocab_size

                    output = output[:, :, :].contiguous().view(-1, output_dim)
                    # output: [batch_size*(target_len - 1), target_vocab_size]

                    target = target[:, :].contiguous().view(-1)
                    # target: [batch_size*(target_len - 1)]

                    loss = self.criterion(output, target)

                    epoch_loss += loss.item()

                test_loss = epoch_loss / len(test)

                print(f"Test loss: {test_loss:.5f}")

                concatenated_tokens = torch.cat(predict_tokens_list, dim=0).tolist()

                list_sentence = [' '.join(self.vocab_en.convert_ids_to_tokens(ids)) for ids in concatenated_tokens]

                # make csv file
                df = pd.DataFrame({"predict": list_sentence,
                                   "ground truth": gt})
                df.to_csv("result.csv", index= False)