model_path=$1
name_keys=$2
path_2_data=$3
lang_list=$4
lang_pairs=$5
GPU=$6
python check_sparsity.py $model_path/checkpoint_last_iter0.pt
target_lang=de
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate $path_2_data \
    --path $model_path/checkpoint_last_iter0.pt \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang en \
    --target-lang $target_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > $model_path/from_en_to_${target_lang}_${name_keys}.txt