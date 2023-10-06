import torch
import os 
import numpy as np 
import matplotlib.pyplot as plt
import math 
import sys


def read_bleu(path):
    with open(path) as f:
        data = f.readlines()
    data = data[-1]
    key = 'Generate test with beam=5: BLEU4 = '
    bleu_score = data[len(key): len(key)+5]
    stop_key = bleu_score.find(',')
    # process_score
    bleu_score = bleu_score[:stop_key]
    bleu_score = float(bleu_score)
    return bleu_score


lang_list = ['fr', 'cs', 'de', 'gu', 'ja', 'my', 'ro', 'ru', 'vi', 'zh']
sparsity = 0
task = '22'
stop_key = 1


overall_data = {}
task_list = ['22', '55', '101', '1010']
stop_key_list = [1,5,10,10]
for task, stop_key in zip(task_list, stop_key_list):
    overall_data[task] = {
        'score_task': [],
        'score_all': [],
        'score_task_reverse': [],
        'score_all_reverse': []
    }
    for sparsity in range(6):

        single_sparse_result = []
        for lang in lang_list:
            path = 'en_' + lang + '_' + task + '_' + str(sparsity) + '.txt'
            bleu = read_bleu(os.path.join('bleu_result', path))
            single_sparse_result.append(bleu)
        single_sparse_result = np.array(single_sparse_result)
        score_task = np.mean(single_sparse_result[:stop_key])
        score_all = np.mean(single_sparse_result)

        single_sparse_result_reverse = []
        for lang in lang_list:
            path = 'en_' + lang + '_reverse_' + task + '_' + str(sparsity) + '.txt'
            bleu = read_bleu(os.path.join('bleu_result', path))
            single_sparse_result_reverse.append(bleu)
        single_sparse_result_reverse = np.array(single_sparse_result_reverse)
        score_task_reverse = np.mean(single_sparse_result_reverse[:stop_key])
        score_all_reverse = np.mean(single_sparse_result_reverse)

        overall_data[task]['score_task'].append(score_task)
        overall_data[task]['score_all'].append(score_all)
        overall_data[task]['score_task_reverse'].append(score_task_reverse)
        overall_data[task]['score_all_reverse'].append(score_all_reverse)


# plot curve 
# for key in ['score_task', 'score_all', 'score_task_reverse', 'score_all_reverse']:
#     for task in task_list:
#         plt.plot(overall_data[task][key], label = task)
#         plt.legend()
#         plt.title(key + task)
#         plt.show()


for i, task in enumerate(task_list):
    plt.subplot(2,2,i+1)
    result = np.array(overall_data[task]['score_task']) + np.array(overall_data[task]['score_task_reverse'])
    plt.plot(result, label = task)
    plt.legend()
    plt.title(task + 'task')
plt.show()

for i, task in enumerate(task_list):
    plt.subplot(2,2,i+1)
    result = np.array(overall_data[task]['score_all']) + np.array(overall_data[task]['score_all_reverse'])
    plt.plot(result, label = task)
    plt.legend()
    plt.title(task + 'all')
plt.show()





