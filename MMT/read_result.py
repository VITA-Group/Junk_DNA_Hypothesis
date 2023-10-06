import os 
import sys 
import numpy as np 
import torch




print('Searching Result from {}'.format(sys.argv[1]))
result = {}
for i in range(11):
    print('Processing Pruning State = {}'.format(i))
    if not os.path.exists(os.path.join(sys.argv[1], str(i))): continue
    path_list = os.listdir(os.path.join(sys.argv[1], str(i)))
    single_sparse_result = {}
    for data_path in path_list:
        if not '.txt' in data_path: continue
        with open(os.path.join(sys.argv[1], str(i), data_path)) as f:
            data_lines = f.readlines()
        last_line = data_lines[-1]
        offest = len('Generate test with beam=5: BLEU4 = ')
        bleu_score = last_line[offest:offest+6]
        clip = bleu_score.find(',')
        bleu_score = float(bleu_score[:clip])

        if not 'from' in data_path:
            if 'reverse' in data_path:
                name_key = 'to_en_from_{}'.format(data_path[3:5])
            else:
                name_key = 'from_en_to_{}'.format(data_path[3:5])
        else:
            name_key = data_path[:13]

        single_sparse_result[name_key] = bleu_score
    result[i] = single_sparse_result

# show result
print('State \t To fr, \t cs, \t, de, \t gu, \t ja, \t my, \t ro, \t ru, \t vi, \t zh')
for i in range(11):
    if not i in result: continue
    data = result[i]
    for lang in ['fr', 'cs', 'de', 'gu', 'ja', 'my', 'ro', 'ru', 'vi', 'zh']:
        if not 'from_en_to_{}'.format(lang) in data:
            data['from_en_to_{}'.format(lang)] = -1
    print('{} \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}'.format(i,
        data['from_en_to_fr'],data['from_en_to_cs'],data['from_en_to_de'],data['from_en_to_gu'],data['from_en_to_ja'],
        data['from_en_to_my'],data['from_en_to_ro'],data['from_en_to_ru'],data['from_en_to_vi'],data['from_en_to_zh']))
print('State \t From fr, \t cs, \t, de, \t gu, \t ja, \t my, \t ro, \t ru, \t vi, \t zh')
for i in range(11):
    if not i in result: continue
    data = result[i]
    for lang in ['fr', 'cs', 'de', 'gu', 'ja', 'my', 'ro', 'ru', 'vi', 'zh']:
        if not 'to_en_from_{}'.format(lang) in data:
            data['to_en_from_{}'.format(lang)] = -1
    print('{} \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}, \t {:.2f}'.format(i,
        data['to_en_from_fr'],data['to_en_from_cs'],data['to_en_from_de'],data['to_en_from_gu'],data['to_en_from_ja'],
        data['to_en_from_my'],data['to_en_from_ro'],data['to_en_from_ru'],data['to_en_from_vi'],data['to_en_from_zh']))
        

torch.save(result, 'BLEU_{}.pt'.format(sys.argv[2]))


















