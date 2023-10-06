model=$1
source_lang=$2
target_lang=$3
path_2_data=$4
lang_list=examples/multilingual/lang_list.txt
lang_pairs=en-fr,fr-en
key=$6

CUDA_VISIBLE_DEVICES=$5 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $source_lang \
    --target-lang $target_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_${key}.txt