from torch.utils.data import Dataset, DataLoader
import os
import lzma
import torch
import random
import string


# preprocessing : https://huggingface.co/docs/transformers/preprocessing
# have batch : https://huggingface.co/docs/transformers/glossary#attention-mask
# train your own tokenizer https://youtu.be/UkNmyTFKriI
# notebook hugging face https://github.com/huggingface/transformers/tree/main/notebooks

def generate_compress_file(dataset_path="./dataset", length=500):
    text_folder = os.path.join(dataset_path, "text")
    compress_folder = os.path.join(dataset_path, "compress")

    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            text_file_path = os.path.join(text_folder, filename)
            compress_file_path = os.path.join(compress_folder, f"{filename}.xz")

            with open(text_file_path, "rb") as text_file:
                data = text_file.read()

            # Compression et écriture des données dans le fichier compressé
            with lzma.open(compress_file_path, "w") as compressed_file:
                compressed_file.write(data)

    print("Compression terminée.")





class simpleDataLoader(Dataset):
    pass

class CompressDataset(Dataset):
    def __init__(self, text_folder="./dataset/text", compress_folder="./dataset/compress"):
        self.text_folder = text_folder
        self.compress_folder = compress_folder
        self.file_list = [file for file in os.listdir(text_folder) if file.endswith(".txt")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        text_file_path = os.path.join(self.text_folder, filename)
        compress_file_path = os.path.join(self.compress_folder, f"{filename}.xz")

        with open(text_file_path, "rb") as text_file:
            text_data = torch.tensor(bytearray(text_file.read()), dtype=torch.uint8)

        with open(compress_file_path, "rb") as compress_file:
            compress_data = torch.tensor(bytearray(compress_file.read()), dtype=torch.uint8)

        return {'text': text_data, 'label': compress_data}


custom_dataset = CompressDataset()
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# Utilisation du DataLoader dans l'entraînement du modèle
for batch in dataloader:
    texts = batch['text']
    labels = batch['label']
    print(type(texts[0]))
    print(type(labels[0]))
    # Faire quelque chose avec le batch (ici, c'est un batch de textes et de labels)
    print("Batch shape (text):", texts.shape)
    print("Batch labels:", labels)
    break
