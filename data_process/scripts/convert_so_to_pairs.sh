
langs=('javascript' 'java' 'ruby' 'php' 'go' 'c++' 'python' 'c#' 'c' 'rust')

for lang in ${langs[@]}; do
    echo "Converting ${lang} pl..."
    python convert_pairs.py $lang
done
