#  [Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs)](https://arxiv.org/pdf/2310.02277.pdf)



[Lu Yin*](https://luuyin.com//), [Ajay Jaiswal*](https://ajay1994.github.io/), [Shiwei Liu](https://shiweiliuiiiiiii.github.io/),[Souvik Kundu](https://ksouvik52.github.io/), [Zhangyang Wang](https://vita-group.github.io/)

University of Texas at Austin, Eindhoven University of Technology, University of Oxford, Intel Labs

The code can be contacted at l.yin@tue.nl.

Table of contents
* [Installation](#installation)
* [Various Task Diffuclty](#various-task-diffculty)
* [Are Pre-trained Magnitude Values Indeed the True Gem?](#are-pre-trained-magnitude-values-indeed-the-true-gem?)

--- 

## Installation 
Please check [INSTALL.md](INSTALL.md) for installation instructions.



## Various Task difficulty


We provide a quick overview of the arguments:  
- `--model_name_or_path`: The identifier for the model on the Hugging Face model hub.
- `--TASK_NAME`: the name of the fine-tuned tasks.
- `--sparsity`: Denotes the percentage of weights to be pruned.
- `--sparse_init`: Specifies the type of sparsity [`sparse_nm`, `sparse_unstuctured`] .
- `--method`: a flag to of the output_dir.



### TASK DIFFICULTY SETTING 2: Varying the Adequacy of Target Domain Data
--- 
### Scripts example

```
TO BE RELEASED SOON
```




### TASK DIFFICULTY SETTING 2: Varying the Option Count in Multiple-choice QA Setting
--- 


### Scripts example


```
TO BE RELEASED SOON
```



### TASK DIFFICULTY SETTING 3: Varying context length for Retrieval-Augmented QA
--- 

```
TO BE RELEASED SOON
```



### Scripts 

in the `./open_closed_book` folder





### Task Difficulty Setting 4: Estimating LLM-facing Task Difficulty by Normalized Human-LLM Performance Gap
--- 

### Scripts example

```
TO BE RELEASED SOON
```


## Are Pre-trained Magnitude Values Indeed the True Gem?

### Scripts example

--
vary  `sparse_method` with 
- `--freeze_weights`: Sparse Transfer
- `--freeze_weights_frompretrain`: Dense Transfer with Freezing
- -or leave it emply:  Sparse to Dense Transfer

```
cd ./GLUE_tasks 

for seed in 41 42 43
do
  for TASK_NAME in QNLI 
  do 
    for sparsity in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
      for validation_split_percentage in 100
      do
      python Glue_prune_oneshot.py \
        --method Glue_noembed_freeze_weights \
        --validation_split_percentage $validation_split_percentage \
        --sparse_method \
        --noembed \
        --sparsity $sparsity \
        --model_name_or_path roberta-large \
        --task_name $TASK_NAME \
        --max_length 512 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --seed $seed \
        --output_dir ./roberta/Glue_noembed_freeze_weights/$TASK_NAME/$sparsity/$validation_split_percentage/$seed/
      done
    done
  done
done


```

### Results
<p align="center">
<img src="./Images_png/TRUE_GEM.png" width="700" height="380">
</p>



## Citation
if you find this repo is helpful, please cite

```
@article{yin2024junk,
  title={Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs},
  author={Yin, Lu and Jaiswal, Ajay and Liu, Shiwei  and Kundu, Souvik and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2310.02277v2},
  year={2024}
}

```
