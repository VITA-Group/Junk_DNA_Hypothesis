
path_2_data=examples/multilingual/multidata
lang_list=examples/multilingual/lang_list.txt
lang_pairs=zh-en,vi-en,ru-en,ro-en,my-en,ja-en,gu-en,de-en,cs-en,fr-en
pretrained_model=examples/multilingual/mbart.cc25.v2/model.pt
save_dir=mling_10_1_true

# # First Training Loop
# CUDA_VISIBLE_DEVICES=$1 python train_custom_new.py "$path_2_data" \
#     --encoder-normalize-before --decoder-normalize-before \
#     --arch mbart_large --layernorm-embedding \
#     --task translation_multi_simple_epoch \
#     --restore-file "$pretrained_model" \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --sampling-method "temperature" \
#     --sampling-temperature "1.5" \
#     --encoder-langtok "src" \
#     --decoder-langtok \
#     --lang-dict "$lang_list" \
#     --lang-pairs "$lang_pairs" \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
#     --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
#     --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
#     --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
#     --max-tokens 1024 --update-freq 2 \
#     --save-interval 1 --save-interval-updates 50000 --keep-interval-updates 10 --no-epoch-checkpoints \
#     --seed 222 --log-format simple --log-interval 100 --save-dir $save_dir/0/ --fix --sparse-init iterative_gm --sparsity 0.2 --imp-iters 0

# # Calculate Scores
# bash multi_lingual/lth/score.sh $save_dir/0/ ${save_dir}_0 $path_2_data $lang_list $lang_pairs $1
# # Delete Best Model to save storage
# rm -rf $save_dir/0/checkpoint_best_iter0.pt

for ((i=5; i<11; i++))
do
    j=$((i-1))
    CUDA_VISIBLE_DEVICES=$1 python train_custom_new.py "$path_2_data" \
        --encoder-normalize-before --decoder-normalize-before \
        --arch mbart_large --layernorm-embedding \
        --task translation_multi_simple_epoch \
        --restore-file $save_dir/$j/checkpoint_last_iter0.pt \
        --reset-optimizer --reset-dataloader --reset-meters \
        --sampling-method "temperature" \
        --sampling-temperature "1.5" \
        --encoder-langtok "src" \
        --decoder-langtok \
        --lang-dict "$lang_list" \
        --lang-pairs "$lang_pairs" \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
        --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
        --max-tokens 1024 --update-freq 2 \
        --save-interval 1 --save-interval-updates 50000 --keep-interval-updates 10 --no-epoch-checkpoints \
        --seed 222 --log-format simple --log-interval 100 --save-dir $save_dir/$i/ --fix --sparse-init iterative_gm --sparsity 0.2 --imp-iters $i
    
    # Calculate Scores
    bash multi_lingual/lth/score.sh $save_dir/$i/ ${save_dir}_$i $path_2_data $lang_list $lang_pairs $1
    # Delete Best Model to save storage
    rm -rf $save_dir/$i/checkpoint_best_iter0.pt
    rm -rf $save_dir/$j/checkpoint_last_iter0.pt
done

rm -rf $save_dir/9/checkpoint_last_iter0.pt
