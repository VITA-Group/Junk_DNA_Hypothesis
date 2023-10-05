from fileinput import filename
import json
import os
import numpy as np
prune_ratio =  ["0.0", "0.05", "0.1", "0.15",  "0.2",  "0.25",  "0.3",  "0.35",  "0.4",  "0.45",  "0.5", "0.55", "0.6", "0.65", "0.7"]
prune_type = "wanda"
def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

gpt_35 = "./answer.jsonl" 
gpt35_data = get_json_list(gpt_35)

gpt35_unique_count = 0
gpt35_word_count = 0
for i in range(0, len(gpt35_data)):    
    gt = gpt35_data[i]["text"]
    gpt35_unique_count += len(set(gt.split(" ")))
    gpt35_word_count += len(gt.split(" "))
print(f"GPT 3.5 Unique Count : {gpt35_unique_count}")
print(f"GPT 3.5 Total Word Count : {gpt35_word_count}")

unique_count = []
total_count = []
for p in prune_ratio:
    file_name = f"./runs/llama7b_free_form/{prune_type}/answer_{p}_unstructured.jsonl"
    pruned_data = get_json_list(file_name)
    
    pruned_unique_count = 0
    pruned_word_count = 0
    for i in range(0, len(pruned_data)):
        pt = pruned_data[i]["text"]
        pruned_unique_count += len(set(pt.split(" ")))
        pruned_word_count += len(pt.split(" "))
    unique_count.append(pruned_unique_count)
    total_count.append(pruned_word_count)
print(f"Unique: {unique_count}")
print(f"Total: {total_count}")

# Opening JSON file
# f = open(file_name)
 
# # returns JSON object as
# # a dictionary
# data = json.load(f)
# print(f"{file_name} data loaded.")