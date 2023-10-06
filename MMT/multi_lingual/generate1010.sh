
model=$1
source_lang=en
path_2_data=examples/multilingual/multidata
lang_list=examples/multilingual/lang_list.txt
lang_pairs=en-fr,en-cs,en-de,en-gu,en-ja,en-my,en-ro,en-ru,en-vi,en-zh,zh-en,vi-en,ru-en,ro-en,my-en,ja-en,gu-en,de-en,cs-en,fr-en
key=$2


target_lang=cs
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=fr
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=de
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=gu
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=ja
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=my
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=ro
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=ru
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=vi
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt


target_lang=zh
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
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
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $target_lang \
    --target-lang $source_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_reverse_${key}.txt

