
# 'ruby' 'python'

langs=('javascript' 'java' 'php' 'go' 'c++' 'c#' 'c' 'rust')

for lang in ${langs[@]}; do
    echo "Decontaminating ${lang} pl..."
    python find_substrings.py --input-path /data01/lizehan/proqa/pls/qa.en.${lang}.json --output-dir so-${lang} --lang $lang
done
