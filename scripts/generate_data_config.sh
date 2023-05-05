

DATA_PATH='/data01/lizehan/proqa/pls'


function count () {
    DATA_DIR=$1;
    for FILENAME in `ls ${DATA_DIR}`; do
        FILEPATH=${DATA_DIR}/${FILENAME}
        LINES=$(wc -l ${FILEPATH} | awk '{print $1}')
        echo $FILENAME $LINES
    done
}


function write () {
    count ${DATA_PATH} | jq -n -R '
        [inputs] | 
        map( split(" ") | .[1] |= tonumber | {
            name: .[0],
            lines: .[1],
        } )'
}

write > config/data_config.json

