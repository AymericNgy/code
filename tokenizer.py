import torch


def tokenizer(byte_tensor, length):
    """

    :param byte_tensor: torch tensor uint8
    :param length: length of the output, complete with padding if shorter
    :return: torch tensor int16
    """
    output = dict()
    new_tensor = (byte_tensor[:length] + 1)
    zero_tensor = torch.zeros(length, dtype=torch.int16)
    zero_tensor[:new_tensor.size(0)] = new_tensor
    output['input_ids'] = zero_tensor
    output['token_type_ids'] = torch.zeros(length, dtype=torch.uint8)
    output['attention_mask'] = zero_tensor > 0
    return output


if __name__ == '__main__':
    length = 10
    byte_tensor = torch.tensor([1, 0, 3], dtype=torch.uint8)
    print(byte_tensor)
    print(tokenizer(byte_tensor, length))
