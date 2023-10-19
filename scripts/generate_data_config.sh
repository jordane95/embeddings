

DATA_PATH=$1 # Path to data directory
CONFIG_PATH=$2 # Path to save data config


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

write > $CONFIG_PATH


