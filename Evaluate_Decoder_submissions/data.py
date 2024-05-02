import torch

def get_data(vocab_size,batch_size,seq_len,embed_dim,seed=1729):
    '''
    Generate decoder token_ids, label_ids and enc_rep for mhca module
    '''
    label_ids = torch.randint(low=1,high=vocab_size-1,size=(batch_size,seq_len-1),generator=torch.random.manual_seed(1947))
    token_ids = torch.zeros(size=(batch_size,seq_len),dtype=torch.int64)
    token_ids[:,1:]=label_ids[:,0:]
    # No gradient is required
    enc_rep = torch.randn(size=(batch_size,seq_len,embed_dim),generator=torch.random.manual_seed(10))
    return (token_ids,label_ids,enc_rep)


