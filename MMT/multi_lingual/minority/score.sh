model_path=$1
name_keys=$2
path_2_data=$3
lang_list=$4
lang_pairs=$5


python check_sparsity.py $model_path/checkpoint_last.pt




target_lang=gu
 fairseq-generate $path_2_data \
    --path $model_path/checkpoint_last.pt \
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
 fairseq-generate $path_2_data \
    --path $model_path/checkpoint_last.pt \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang en \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > $model_path/to_en_from_${target_lang}_${name_keys}.txt

