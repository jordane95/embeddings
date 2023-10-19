
langs=('javascript' 'java' 'ruby' 'php' 'go' 'cpp' 'python' 'c_sharp' 'c' 'rust')

for lang in ${langs[@]}; do
    echo "Converting ${lang} pl..."
    python convert_valut_pairs.py $lang
done
