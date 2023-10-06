#  [Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity)](https://arxiv.org/pdf/2310.02277.pdf)

Official PyTorch implementation of  **Junk DNA Hypothesis**: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity 

[Lu Yin](https://luuyin.com//), [Shiwei Liu](https://shiweiliuiiiiiii.github.io/), [Ajay Jaiswal](https://ajay1994.github.io/), [Souvik Kundu](https://ksouvik52.github.io/), [Zhangyang Wang](https://vita-group.github.io/)

University of Texas at Austin, Eindhoven University of Technology

The code can be contacted at l.yin@tue.nl.

Table of contents
* [Installation](#installation)
* [Various Task Diffuclty](#various-task-diffculty)
* [Are Pre-trained Magnitude Values Indeed the True Gem?](#are-pre-trained-magnitude-values-indeed-the-true-gem?)

--- 

## Installation 
Please check [INSTALL.md](INSTALL.md) for installation instructinos.



## Various Task diffculty


We provide a quick overview of the arguments:  
- `--model_name_or_path`: The identifier for the model on the Hugging Face model hub.
- `--TASK_NAME`: the name of the fine-tuned tasks.
- `--sparsity`: Denotes the percentage of weights to be pruned.
- `--sparse_init`: Specifies the type of sparsity [`sparse_nm`, `sparse_unstuctured`] .
- `--method`: a flag to of the output_dir.



### Task Difficulty Setting 1: Varying the Adequacy of Target Domain Data
--- 
### Scripts example

```
cd ./GLUE_tasks
for seed in 41 
do
  for TASK_NAME in qnli  
  do 
    for sparsity in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
      for validation_split_percentage in 10 25 50 75 100
      do
      python Glue_prune_oneshot.py \
        --method Glue_noembed_freeze_weights \
        --validation_split_percentage $validation_split_percentage \
        --freeze_weights \
        --noembed \
        --sparsity $sparsity \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --max_length 128 \
        --per_device_train_batch_size 32 \
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
<img src="./Images_png/Task_1.png" width="700" height="400">
</p>



### Task Difficulty Setting 2: Majority v.s. Minority in Multi-Domain Learning
--- 


### Scripts example


```

cd ./MMT
path_2_data=path/to/multidata
lang_list=examples/multilingual/lang_list.txt
lang_pairs=en-ru,en-vi,vi-en,ru-en
pretrained_model=path_2_data=path/to/pretrained_model
save_dir=majority_pre_2to2


for sparsity in 0.8
do
    python train.py "$path_2_data" \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --restore-file "$pretrained_model" \
    --reset-optimizer --reset-dataloader --reset-meters \
    --sampling-method "temperature" \
    --sampling-temperature "1.5" \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 1 --max-update 1 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --save-interval 1 --save-interval-updates 50000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --sparse --log-format simple --log-interval 100 --save-dir $save_dir/$sparsity/ --fix --sparse-init one_shot_gm_cpu --sparsity $sparsity
    bash multi_lingual/majority_pre/score.sh $save_dir/$sparsity/ ${save_dir}_$sparsity $path_2_data $lang_list $lang_pairs 
    # Delete Best Model to save storage
    rm -rf $save_dir/$sparsity/checkpoint_best.pt
    rm -rf $save_dir/$sparsity/checkpoint_last.pt
    rm -rf $save_dir/$sparsity/checkpoint_last.pt
done
```

### Results

<p align="center">
<img src="./Images_png/Task_2.png" width="700" height="180">
</p>



### Task Difficulty Setting 3: With v.s. Without Available External Information
--- 



### Scripts 

in the `./open_closed_book` folder



### Results
<p align="center">
<img src="./Images_png/Task_3.png" width="700" height="200">
</p>


### Task Difficulty Setting 4: Estimating LLM-facing Task Difficulty by Normalized Human-LLM Performance Gap
--- 

### Scripts example

```
cd ./GLUE_tasks
for seed in 41 42 43
do
  for TASK_NAME in cola sst2
  do 
    for sparsity in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
      for validation_split_percentage in 100
      do
      python Glue_prune_oneshot.py \
        --method Glue_noembed_freeze_weights \
        --validation_split_percentage $validation_split_percentage \
        --freeze_weights \
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
<img src="./Images_png/Task_4.png" width="700" height="350">
</p>

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

```
