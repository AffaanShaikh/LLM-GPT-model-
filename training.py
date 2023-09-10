import torch 
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description = 'This is a demonstration')
#Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help="pls provide batchsize")
args = parser.parse_args() #parse the arguments

#now we may use the argument value in our program
print(f'batch_size:{args.batch_size}')

#if gpu is available set device to cuda else cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#block size is size of blocks within the training data. batch_size is how many blocks are done in parallel
#an Auto-Tuning script to find optimal parameters for a specific machine
block_size = 12 #block size/ sequence length 
batch_size = args.batch_size #how many blocks we want at the same time
max_iters = 1000 #how many iterations we want to do
learning_rate = 1e-4 #more learning rates -> 3e-3, 3e-4, 1e-3, 1e-4
eval_iters = 200 #reporting the loss
n_embd = 260 #parameter denoting the length of the vector holding attribute about tokens OR 
             #the total dimensions captured from all the concatenated heads
n_head = 3 # no. of heads we have running 
n_layer = 3 #no. of layers/blocks(decoder blocks)
dropout = 0.2 #dropping 20% of neurons from the neural net

#Reading the file and visualizing first 200 words
chars = ""
with open('X:/LLM/llm-course/openwebtext/vocab_txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
#print(text[:100])
#What characters are in the book and the total number of them are printed
#A Vocabulary set of words and symbols
#print(chars)
vocab_size = len(chars)
#print(vocab_size)

#dictionaries for string to int and vice-versa, ch - character, i - char's index value i.e. equivalent int. value
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
#string 's' is input
default_integer = 2002
encode = lambda s: [string_to_int[c] for c in s] 
#to handle special characters that are not present in string_to_int
encode = lambda s: [string_to_int.get(c, default_integer) for c in s]
#list of integers 'l' is input
#.join to concatenate the encoded integers(i.e. characters) back into a string
decode = lambda l: ''.join([int_to_string[i] for i in l]) 

#memort mapping: a way of looking at disk files, opening and looking at them without 
#having to open the entire thing at once. Looks at little chunks at a time in a large text file
# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "openwebtext/output_train.txt" if split == 'train' else "openwebtext/output_val.txt"
    with open(filename, 'rb') as f: #opening in 'rb' binary mode
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            #from 0 to filesize(complete length) - the margin i.e. (block_size*batch_size)
            start_pos = random.randint(0, (file_size) - block_size*batch_size) 

            # Seek(go up to) to the random position and read the block of text that has size of (block_size*batch_size-1)
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences. Need to decode from 'rb' binary format to utf-8
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '') #replacing '\r'(errors) with a blank space
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #print(ix)
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#a, b = get_batch('train')
#print('inputs:')
#print(a)
#print('targets:')
#print(b)

#decorator to ensure pytorch doesn't use gradients at all, reduces computation and memory usage, better performance.
#gradients used during training, no training done here so...no_grad()
#function for reporting and evaluating how the model is performing
@torch.no_grad()
def estimate_loss():
    out = {}
    #model set to evaluation mode, dropout disabled 
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #model set to training mode, dropout enabled 
    model.train()
    return out

class Head(nn.Module):
    #one head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #.register_buffer registers the nolookahead masking to the model state,
        #no need to reinitialize the heads for every forward and backward pass, saving computation and training time
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #Input of size (batch, time-step, channels), Output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x) #(B,T,hs) hs->headsize
        q = self.query(x) #(B, T, hs)
        #compute attention scores("affinities")
        #.transpose sets up a timestable format, allowing us to multiply and flipping -2dim with -1dim(last dim)
        #k.shape[-1]**-0.5 is the scaling done to enable us to distinct and scaled down head size dim 
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) Dot Producting
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) masked_fill to prevent lookahead
        #all 0 converted to '-inf' for softmax to exponent
        #Making model more confident in its predictions by assigning and highlighting appropriate attention
        #when value/attention score is big we want the model to focus more on that and concurrently learn more from it.
        wei = F.softmax(wei, dim=-1) #(B, T, T) 
        wei = self.dropout(wei)
        #perform the weighted aggregation of the values, i.e. adding Values to Keys and Queries final Output 
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    #multiple heads of self-attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        #ModuleList -> Holds submodules in a list, essentially having "Head"s in parallel for each head(num_heads)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #.proj ->  transforms the concatenated output from a size of (head_size * num_heads) to a size of n_embd.
        #purpose is to combine information from all the heads and transform it into a form that can be used by the subsequent layers or models
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #concatenating each head together along the last dimension(i.e. the C/Features, last dim)
        out = torch.cat([h(x) for h in self.heads], dim=-1) #dim -1 meaning in C/Features of (B, T, C/Features)
        out = self.dropout(self.proj(out))
        return out

        
class FeedForward(nn.Module):
    #a simple linear layer followed by a non-linearity

    def __init__(self, n_embd):
        super().__init__()
        #Sequential “chains” outputs to inputs sequentially for each subsequent module
        #Output of .Linears/output shape will be of dimension (n_embd by n_embd). Also making sure inner dimensions line up(i.e. 4*n_embd)
        #ReLU -> introduces non-linearity by outputting the input directly if it's positive, and zero otherwise
        #dropout ->  randomly drop units (along with their connections) from the neural network during training, prevents overfitting.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
#Transformer block: communication followed by computation

    def __init__(self, n_embd, n_head):
        super().__init__()
        #n_embd: embedding dimension, n_head: the no. of heads we'd like
        #head_size: no. of features each head will be capturing in MultiHeadAttention
        head_size = n_embd // n_head
        #self-attention
        self.sa = MultiHeadAttention(n_head, head_size)
        #feed-forward
        self.ffwd = FeedForward(n_embd)
        #2 layer normalizations
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        #DECODER architecture -> selfattention, add a norm, feedforward, add a norm
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

#logits are bunch of floating point numbers that are normalized
#they are probability distributions of what we want to predict
#normalization is the division of single element with sum of all elements
#B is Batch, T is Time dim.(value we don't know), C is Channels/how many are there or the vocab size.
#nn.Module works as a tracker for your parameter and ensure nn extentions work correctly
class GPTLanguageModel(nn.Module): #GPT - Generative Pre-trained Transformer
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #.blocks is how many blocks/layers we have running sequentially
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm, added at the end of decoders
        self.lm_head = nn.Linear(n_embd, vocab_size) #final linear transformation nn.Linear, to work with softmax

        #making sure weights are initialized properly, if standard deviation is too high, outliers occur and 
        #learning is difficult, if it's low, no learning is done, no oppurtunity for varied parameters to learn
        self.apply(self._init_weights)

    #initializing weights around certian standard deviations
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        #logits = self.token_embedding_table(index)
        B, T = index.shape #unpacking B and T
        
        #index and targets are both (B,T) tensor of integers
        #Broadcasting Semantics - Rules about operations done on Tensors
        tok_emb = self.token_embedding_table(index) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)  
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C) #finallayernorm
        logits = self.lm_head(x) #(B,T,vocab_size) #nn.linearTransformation before softmax

        if targets is None:
            loss = None
        else:
            #blending B and T since C is what is being observed and
            #as long as logits and targets have same B and T, should be fine
            #since doc says parameter to be in the form (N, C) where N is our Batch(No. of batches)
            #i.e. in the form -> (B*T(made into 1), C) instead of B, T, C
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss    

    def generate(self, index, max_new_tokens):
        #index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self.forward(index)
            # focusing only on the last time step
            logits = logits[:, -1, :] #(B, C)
            #applying softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #append sample index to the running sequence 
            index = torch.cat((index, index_next), dim=1) #(B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
#make sure there is a model to from before pickling or else errors
print("loading model parameters...")
with open('model-01.pkl', 'rb') as f: #'rb' -> reading in binary
    model = pickle.load(f)
print('loading done')
m = model.to(device)
#torch.long same as int64

#context = torch.zeros((1,1), dtype=torch.long, device=device)
#generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
#print(generated_chars)

#Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step:{iter}, train loss:{losses['train']:.3f}, val loss:{losses['val']:.3f}")
    #sampling a batch of data
    #xb-inputs, yb-targets
    #evaluating the loss
    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    #.zero_grad usually used in RNN
    optimizer.zero_grad(set_to_none=True)#set to none instead of zero gradient accumulated by Pytorch
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f: #'wb' -> to write in binary
    pickle.dump(model, f) #saving
print('Model has been saved!')