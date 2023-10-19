
langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')

for lang in ${langs[@]}; do
    echo "Converting ${lang} pl..."
    python convert_csn.py $lang
done

