import numpy as np
import warnings
import json
from tqdm import tqdm
import argparse
import os
import cv2
import random
import torch
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data_json', type=str, required=True, help='Path to the input JSON file containing video questions')
parser.add_argument('--video_path', type=str, default='./data', help='Path to the video directory')
parser.add_argument('--ckpt', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Path to model checkpoints')
parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample.")
parser.add_argument("--sample_fps", type=float, default=0.0, help="Sample fps (0 means disable)")
parser.add_argument("--save_name", type=str, default="", help="Save name")

# Parse arguments
args = parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set seed for all available GPUs

##################### Initilaize the model #####################
warnings.filterwarnings("ignore")

def get_video_duration_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt, 
        torch_dtype=torch.bfloat16,   
        attn_implementation="flash_attention_2",
        device_map="auto" 
    )
    base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return tokenizer, model, processor

##################### Get response #####################
@torch.no_grad()
def get_response(model, tokenizer, processor, video_path, nframes, query):
    assert os.path.exists(video_path), f"Video file {video_path} does not exist."
    video_duration = get_video_duration_opencv(video_path)
    if video_duration == 0:
        print(f'Video {video_path} has zero duration, skipping...')
        return 'Error'
    fps = min(args.sample_fps, nframes / video_duration) if args.sample_fps > 0 else nframes / video_duration
    video_info = {"type": "video", "video": video_path, "fps": fps}
    messages = [
        {
            "role": "user",
            "content": [
            video_info,
            {"type": "text", "text": query},
        ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")
    # type same as model
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)
    
    temperature = 0
    do_sample = False
    generated_ids = model.generate(**inputs, temperature=temperature, do_sample=do_sample, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    print(f"Response: {response}")
    return response


def main(data_json):
    set_seed(42)
    tokenizer, model, image_processor = load_model()
    with open(data_json, 'r') as f:
        questions = json.load(f)

    save_path = data_json.replace('input/', 'output/').replace('.json', '')
    save_name = os.path.basename(args.ckpt)
    if 'checkpoint' in save_name:
        save_name = args.ckpt.split('/')[-2] + '_' +  save_name
    fps_name = f"{args.sample_fps}fps_" if args.sample_fps > 0 else "_"
    text_ans_file_path = os.path.join(save_path, f'{args.nframes}frames{fps_name}{save_name}{args.save_name}.jsonl')
    os.makedirs(os.path.dirname(text_ans_file_path), exist_ok=True)
    print(f'Output file: {text_ans_file_path}')
    
    existing_responses = {}
    if os.path.exists(text_ans_file_path):
        with open(text_ans_file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                existing_responses[data['idx']] = data['response']
    text_ans_file = open(text_ans_file_path, 'a')

    for question in tqdm(questions):
        if question["qa_idx"] in existing_responses:
            continue  
        video_path = os.path.join(args.video_path, question["video_name"])
        response = get_response(model, tokenizer, image_processor, video_path, args.nframes, question["question"])
        text_ans_file.write(json.dumps(dict(idx=question["qa_idx"], response=response)) + '\n')
        text_ans_file.flush()
    text_ans_file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_json)