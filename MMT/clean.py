import os 
import sys
import torch


for i in range(6):
    path = os.path.join(sys.argv[1], str(i), 'checkpoint_best_iter0.pt')
    print(path)
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint['model']
    torch.save(checkpoint, path)









