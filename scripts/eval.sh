DATA_FILES=(
    "data/data_files/input/ReverseFilm.json"
    "data/data_files/input/UCF101.json"
    "data/data_files/input/Rtime_t2v.json"
    "data/data_files/input/Rtime_v2t.json"
    "data/data_files/input/AoTBench_QA.json"
) 

CKPTS=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "sherryxzh/ArrowRL-Qwen2.5-VL-7B"
)

for DATA_FILE in "${DATA_FILES[@]}"; do
    for CKPT in "${CKPTS[@]}"; do
        echo "Evaluating $CKPT on $DATA_FILE"
        python eval/run_qwen25.py --data_json "$DATA_FILE" --ckpt $CKPT
    done
done