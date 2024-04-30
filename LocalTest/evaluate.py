import glob
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_token_ids
from yaml import safe_load
import ray

if ray.is_initialized:    
    ray.shutdown()
ray.init(logging_level=logging.ERROR)

# Load input configuration
with open('enc_config.yml', 'r') as file:
    config = safe_load(file) 

num_configurations = 6
start = time.time()

for path in glob.glob('submissions/*.py'):
  # get the roll number from the path
  roll_number = path.split('/')[1][0:-3]  
  
  # copy the implementation from the file to transformer.py
  shutil.copy2(path,'transformer.py')

  # objects for parallel execution for different configurations
  ray_tasks = [] 

  # import student definitions
  try:
    print(f' Loading {roll_number} implementation..')
    from transformer import MHA, FFN, OutputLayer
  except Exception as err:
    print(err.args)
    continue


  # build an encoder layer
  class EncoderLayer(nn.Module):

    def __init__(self,dmodel,dq,dk,dv,d_ff,heads):
      super(EncoderLayer,self).__init__()
      self.mha = MHA(dmodel,dq,dk,dv,heads)
      self.layer_norm_mha = torch.nn.LayerNorm(dmodel)
      self.layer_norm_ffn = torch.nn.LayerNorm(dmodel)
      self.ffn = FFN(dmodel,d_ff,layer=0)

    def forward(self,x):    
      out = self.layer_norm_mha(self.mha(x)+x)    
      out = self.layer_norm_ffn(self.ffn(out)+out)
      return out

  class Encoder(nn.Module):

    def __init__(self,vocab_size,embed_dim,dq,dk,dv,d_ff,heads,num_layers=1):
      super(Encoder,self).__init__()
      self.vocab_size = vocab_size
      self.embed_dim = embed_dim
      self.embed_weights = nn.Parameter(torch.randn(size=(vocab_size,embed_dim),generator=torch.random.manual_seed(50)))
      self.enc_layers = EncoderLayer(embed_dim,dq,dk,dv,d_ff,heads)
      self.out_layer = OutputLayer(embed_dim,vocab_size)

    def forward(self,x):
      '''
      The input should be tokens ids of size [BS,T]
      '''
      x = self.get_embeddings(x)
      out = self.enc_layers(x)
      out = self.out_layer(out)
      return out


    def get_embeddings(self,x):    
      embed = nn.Embedding(self.vocab_size,self.embed_dim,_weight=self.embed_weights)
      return embed(x)

  @ray.remote(num_cpus=10)
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
      self.token_ids = get_token_ids(vocab_size,batch_size,seq_len)
      self.model = Encoder(vocab_size,self.dmodel,self.dq,self.dk,self.dv,self.d_ff,self.heads)
      self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
      self.criterion = nn.CrossEntropyLoss()

    def train(self):    
      for epoch in range(self.epochs):
        out = self.model(self.token_ids)    
        loss = self.criterion(torch.swapdims(out,dim0=-2,dim1=-1),self.token_ids)
        loss.backward()   
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_accuracy(self):
      self.train()    
      with torch.inference_mode():
          predictions = torch.argmax(self.model(self.token_ids),dim=-1)
      num_correct_predictions = torch.count_nonzero(self.token_ids==predictions)
      accuracy = num_correct_predictions/self.token_ids.numel()
      return accuracy.item() 
  
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
    ['1',32,10,30,8,32,30],
    ['2',32,10,30,8,32,60],
    ['3',32,10,30,8,32,90],
    ['4',64,100,30,80,64,30],
    ['5',64,100,30,80,64,60],
    ['6',64,100,30,80,64,90],
  ]     
  )  

 #get the results
  try:
    print('Running models..')
    results = ray.get(ray_tasks)
  except Exception as err:      
    file_name = f'Report/{str(roll_number)}.txt'
    print('OOPs, there is an error..\nGenerating the report..')
    with open(file_name,'w') as file:
      file.write(str(err))    
    ray.shutdown()

  else:
    file_name = f'Report/{str(roll_number)}.txt'    
    table.add_column('Accuracy',results)
    with open(file_name,'w') as file:
      print('Generating report..')
      file.write(table.get_string())    
    ray.shutdown()    
  end = time.time()
  elapsed = end-start