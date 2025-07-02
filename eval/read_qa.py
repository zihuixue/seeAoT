import glob
import json


def read_qa(input_file):
    with open(input_file, 'r') as f:
        data_list = json.load(f)
    ans_dict = {d['qa_idx']: d['ans'] for d in data_list}
    response_files = glob.glob(f"{input_file.replace('input', 'output').replace('.json', '')}/*.jsonl")
    for response_file in response_files:
        print(f"Evaluating {response_file}")
        correct, total = 0, 0
        with open(response_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                ans = ans_dict[data['idx']]
                pred = data['response'][0]
                correct += (ans == pred)
                total += 1
        print(f"Accuracy: {correct} / {total} = {correct / total:.2%}")


if __name__ == "__main__":
    input_files = glob.glob(f"data/data_files/input/*.json")
    for file in input_files:
        print(f"Processing {file}")
        read_qa(file)
        print("=" * 60)