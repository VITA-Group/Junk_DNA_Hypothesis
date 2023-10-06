import torch
import os 
import sys 


model = torch.load(sys.argv[1], map_location='cpu')
if 'model' in model:
    model = model['model']

zero_weight_cnt = 0
total_weight_cnt = 0

for name in model.keys():
    para = model[name]
    zero_weight_cnt += para.eq(0).float().sum()
    total_weight_cnt += para.nelement()

print('*** Sparsity for {} = {:.4f}%'.format(sys.argv[1], 100 * zero_weight_cnt / total_weight_cnt))







