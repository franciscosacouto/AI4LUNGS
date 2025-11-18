import torch 
torch.set_printoptions(threshold=float('inf'))

data = torch.load("nlst_lung_ct_embeddings_protocol_5_2d_lung.pt")
# Inspect keys
# print(data.keys())
print(type(data))
# Example
embeddings = data["100012"]
print(type(embeddings))     # torch.Tensor
print(embeddings.shape)  
print (embeddings)  