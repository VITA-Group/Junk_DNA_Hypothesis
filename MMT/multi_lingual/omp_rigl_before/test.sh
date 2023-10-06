
path_2_data=examples/multilingual/multidata
lang_list=examples/multilingual/lang_list.txt
lang_pairs=en-fr,en-cs,cs-en,fr-en
pretrained_model=examples/multilingual/mbart.cc25.v2/model.pt
save_dir=omp_rigl_before_2to2
sparsity=$2
state=$3

# First Training Loop
CUDA_VISIBLE_DEVICES=$1 python train.py "$path_2_data" \
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
    --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 200 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --save-interval 1 --save-interval-updates 50000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --sparse --log-format simple --log-interval 100 --save-dir $save_dir/$state/ --sparse-init one_shot_gm_cpu --sparsity $sparsity \
    --sparse-mode DST --prune magnitude --prune-rate 0.5 --growth gradient --update-frequency 5 --redistribution none
