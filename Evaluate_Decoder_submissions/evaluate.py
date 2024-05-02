import shutil
import glob
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data
from yaml import safe_load
import ray
import copy

# Load input configuration
with open('dec_config.yml', 'r') as file:
    config = safe_load(file) 

num_configurations = 6

for path in glob.glob('submissions/*.py'):
  # get the roll number from the path
  roll_number = path.split('/')[1][0:-3]  
  
  # copy the implementation from the file to transformer.py
  shutil.copy2(path,'transformer.py')

  # list of ray objects for parallel execution
  ray_tasks = []

  # import student definitions
  try:
    print(f' Loading {roll_number} implementation..')
    from transformer import MHMA,MHCA,FFN,OutputLayer,PredictionHead
  except Exception as err:
    print(f'Registering error..')
    file_name = f'Report/{str(roll_number)}.txt'
    with open(file_name,'w') as file:
      file.write(str(err.args))
    continue

  # build an decoder layer
  class DecoderLayer(nn.Module):

    def __init__(self,dmodel,dq,dk,dv,d_ff,heads,seq_len,mask=None):
      super(DecoderLayer,self).__init__()
      self.mhma = MHMA(dmodel,dq,dk,dv,heads,seq_len,mask)
      self.mhca = MHCA(dmodel,dq,dk,dv,heads)
      self.layer_norm_mhma = torch.nn.LayerNorm(dmodel)
      self.layer_norm_mhca = torch.nn.LayerNorm(dmodel)
      self.layer_norm_ffn = torch.nn.LayerNorm(dmodel)
      self.ffn = FFN(dmodel,d_ff)

    def forward(self,enc_rep,dec_rep):
      out = self.mhma(dec_rep)
      out = self.layer_norm_mhma(out+dec_rep)      
      out = self.layer_norm_mhca(out+self.mhca(enc_rep,out))      
      out = self.layer_norm_ffn(self.ffn(out)+out)
      return out
  

  class Embed(nn.Module):

    def __init__(self,vocab_size,embed_dim):
      super(Embed,self).__init__()
      embed_weights= torch.randn(size=(vocab_size,embed_dim),generator=torch.random.manual_seed(70))
      self.embed= nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,_weight=embed_weights)

    def forward(self,x):
      out = self.embed(x)
      return out
    
  class Decoder(nn.Module):

    def __init__(self,vocab_size,embed_dim,dmodel,dq,dk,dv,d_ff,heads,seq_len,mask,num_layers=1):
      super(Decoder,self).__init__()
      self.embed_lookup = Embed(vocab_size,embed_dim)
      self.dec_layers = nn.ModuleList(copy.deepcopy(DecoderLayer(dmodel,dq,dk,dv,d_ff,heads,seq_len,mask)) for i in range(num_layers))
      self.predict = PredictionHead(dmodel,vocab_size)

    def forward(self,enc_rep,tar_token_ids):
      dec_rep = self.embed_lookup(tar_token_ids)
      for dec_layer in self.dec_layers:
        dec_rep = dec_layer(enc_rep,dec_rep)
      out = self.predict(dec_rep)

      return out
    
  @ray.remote(num_cpus=1)
  class Trainer(object):

    def __init__(self,vocab_size,batch_size,seq_len,embed_dim,dmodel,n_heads,epochs):
      self.vocab_size = vocab_size
      self.batch_size = batch_size
      self.seq_len = seq_len
      self.heads = n_heads
      self.dmodel = dmodel      
      self.dq = torch.tensor(int(dmodel/n_heads))
      self.dk = torch.tensor(int(dmodel/n_heads))
      self.dv = torch.tensor(int(dmodel/n_heads))
      self.d_ff = torch.tensor(int(4*dmodel))
      self.embed_dim = embed_dim
      self.epochs = epochs
      self.token_ids,self.label_ids,self.enc_rep = get_data(vocab_size,batch_size,seq_len,self.embed_dim)
      self.model = Decoder(self.vocab_size,self.embed_dim,self.dmodel,self.dq,self.dk,self.dv,self.d_ff,self.heads,seq_len,mask=None)
      self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
      self.criterion = nn.CrossEntropyLoss()

    def train(self):   
      self.start = time.time()
      for epoch in range(self.epochs):
        out = self.model(self.enc_rep,self.token_ids)   
        loss = self.criterion(torch.swapdims(out[:,0:-1,:],dim0=-2,dim1=-1),self.label_ids)
        loss.backward()   
        self.optimizer.step()
        self.optimizer.zero_grad()
      self.end = time.time()

    def get_accuracy(self):
      self.train()    
      with torch.inference_mode():
          predictions = torch.argmax(self.model(self.enc_rep,self.token_ids),dim=-1)
      num_correct_predictions = torch.count_nonzero(self.label_ids==predictions[:,0:-1])
      accuracy = num_correct_predictions/self.token_ids.numel()
      return (accuracy.item(),self.end-self.start) # (acc,exec_time)
  
  for i in range(num_configurations):
      
      vocab_size = config['input']['vocab_size'][i]
      batch_size = config['input']['batch_size'][i]
      seq_len = config['input']['seq_len'][i]      
      embed_dim = config['input']['embed_dim'][i]
      dmodel = config['model']['d_model'][i]
      n_heads = config['model']['n_heads']
      epochs = config['train']['epochs'][i]
      model = Trainer.remote(vocab_size,batch_size,seq_len,embed_dim,dmodel,n_heads,epochs) 
      ray_tasks.append(model.get_accuracy.remote())

  # configurations for reporting
  from prettytable import PrettyTable
  table = PrettyTable()
  table.field_names = ["Config",'dmodel',"Vocab_size","Batch_size", "Seq_len","Embed_dim", "Epochs" ]
  table.add_rows([ 
    ['1',32,12,30,8,32,30],
    ['2',32,12,30,8,32,60],
    ['3',32,12,30,8,32,90],
    ['4',64,102,30,80,64,30],
    ['5',64,102,30,80,64,60],
    ['6',64,102,30,80,64,90],
  ]     
  )  
  #get the results
  try:
    print('Running the model..')
    results = ray.get(ray_tasks)
  except Exception as err:
    # report_dict[roll_number]=err.args   
    file_name = f'Report/{str(roll_number)}.txt'
    with open(file_name,'w') as file:
      file.write(str(err))    
    ray.shutdown()

  else:
    file_name = f'Report/{str(roll_number)}.txt'      
    table.add_column('Accuracy',[result[0] for result in results])
    table.add_column('Exec. Time',[result[1] for result in results])
    print('Generating the report..')    
    with open(file_name,'w') as file:
      file.write(table.get_string())    
    ray.shutdown()    
  
  
