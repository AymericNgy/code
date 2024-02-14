import numpy as np
import random
import string
from tqdm import tqdm
import torch

def string_to_hex(input_string):
    # Encode la chaîne en bytes avec UTF-8 et convertit les bytes en une représentation hexadécimale
    hex_representation = input_string.encode('utf-8').hex()
    return hex_representation

# Exemple d'utilisation
chaine = "Hello, World!"
hex_representation = string_to_hex(chaine)
print(f"Chaîne d'origine: {chaine}")
print(f"Représentation hexadécimale: {hex_representation}")


def tuple_to_hex(input_tuple):
    hex_representation = tuple(
        map(lambda x: hex(ord(x))[2:] if isinstance(x, str) else hex(x)[2:], input_tuple)
    )
    return hex_representation

# Exemple d'utilisation
mon_tuple = (0, 0, 'a')
hex_representation_tuple = tuple_to_hex(mon_tuple)
print(f"Tuple d'origine: {mon_tuple}")
print(f"Représentation hexadécimale: {hex_representation_tuple}")

def tuple_to_hex_tensor(input_tuple):
    hex_representation = np.array(
        [int(x, 16) for x in tuple_to_hex(input_tuple)],
        dtype=np.uint8
    )

    tensor_representation = torch.tensor(hex_representation).to('cuda')
    return tensor_representation

# Utilisation de la fonction précédente
mon_tuple = (0, 0, 'a')
hex_representation_tensor = tuple_to_hex_tensor(mon_tuple)
print(f"Tuple d'origine: {mon_tuple}")
print(f"Représentation hexadécimale: {hex_representation_tensor}")



def hex_to_tensor(hex_data):

    paires_hex = [hex_data_decoded[i:i+2] for i in range(0, len(hex_data_decoded), 2)]

    # Convertir chaque paire hexadécimale en un octet
    octets = [int(pair, 16) for pair in paires_hex]

    # Créer un tableau NumPy à partir de la liste d'entiers
    tensor_data = torch.tensor(octets, dtype=torch.float).cuda()

    return tensor_data



def generate_random_text(size_mb, chunk_size_kb=10):
    size_bytes = size_mb * 1024**2
    chunk_size_bytes = chunk_size_kb * 1024
    data = ""

    with tqdm(total=size_bytes, unit='B', unit_scale=True, desc='Generating') as pbar:
        for _ in range(0, size_bytes, chunk_size_bytes):
            chunk_size = min(chunk_size_bytes, size_bytes - len(data))
            random_text = ''.join(random.choice(string.ascii_lowercase + ' ') for _ in range(chunk_size))
            data += random_text
            pbar.update(chunk_size)

    return data

# Exemple d'utilisation pour générer un texte de 50 Mo et l'écrire dans un fichier
