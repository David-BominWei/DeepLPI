# %%
datafile = '/home/wbm001/DeepLPI/DeepLPI/data/seq_dict.csv'
datapath = "/home/wbm001/DeepLPI/DeepLPI/data/"
modelpath = '/home/wbm001/DeepLPI/data/saved_models/prose_mt_3x1024.sav'

# %%
import pandas as pd
data = pd.read_csv(datafile)
data = data[["0","ki_test","kd_test"]]
data

# %%
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from prose.models.multitask import ProSEMT # use the path of ProSE
from prose.alphabets import Uniprot21

model = ProSEMT.load_pretrained(modelpath)
model.eval()
model = model.cuda()

def embed_sequence(x):
    n = model.embedding.proj.weight.size(1)

    alphabet = Uniprot21()
    x = x.upper()
    x = alphabet.encode(x)
    x = torch.from_numpy(x)

    print("encode down")

    x = x.cuda()

    print("loaded to cuda")

    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        z = z.sum(1)
        z = z.cpu().numpy()

    return z

# %%
import numpy as np

print("start to embed")

embed = np.array(list(data["0"].map(lambda x : embed_sequence(x.encode())))).reshape(-1,6165)

print("embed down, start to write file")

# %%
import csv
 
with open(datapath + 'seq_embed.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(embed.tolist())


