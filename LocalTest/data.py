import torch
from torch import Tensor

def get_token_ids(vocab_size,batch_size,seq_len,seed=1729):
    token_ids = torch.randint(low=0,high=vocab_size-1,size=(batch_size,seq_len),generator=torch.random.manual_seed(1729))
    return token_ids