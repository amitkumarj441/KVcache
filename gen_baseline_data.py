import os
import json
import torch
import numpy as np
import multiprocessing as mp
from transformers import AutoTokenizer, Qwen3ForCausalLM
from tqdm import tqdm
import time

MODEL_NAME = "/amit/hf_download/models/Qwen3-4B"
DATA_DIR = "data"
os.makedirs("outputs", exist_ok=True)
OUTPUT_FILE = "data/qwen_8k_2k/baseline.jsonl"
WORKER_BATCH_SIZE = 8

def load_all_data(directory: str) -> list:
    """Load data from all .jsonl files in the specified directory."""
    all_data = []
    print(f"[*] '{directory}' Folder loading data...")
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Warning: Skip to {filepath} Invalid row in: {line}")
    print(f"[*] Loading complete, total {len(all_data)} piece of data。")
    return all_data

def inference_worker(gpu_id: int, data_chunk: list, model_name: str, result_queue: mp.Queue):
    """
    Runs inference on the specified GPU. (This function does not need to be modified.)
    
    Args:
        gpu_id (int): (0, 1, 2, ...)。
        data_chunk (list): This process needs to process a subset of data.。
        model_name (str): model path。
        result_queue (mp.Queue): Process-safe queue for returning results。
    """
    device = f"cuda:{gpu_id}"

    if gpu_id == 0:
        print(f"[Worker-{gpu_id}] It has been started and will be available soon. {device} Run on it. (Other workers will start silently.)")

    try:
        # 1. Load the model on the specified GPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # 2. Perform batch inference on the data assigned
        for i in range(0, len(data_chunk), WORKER_BATCH_SIZE):
            batch = data_chunk[i:i + WORKER_BATCH_SIZE]
            
            prompts = [item['question'] for item in batch]
            texts = []
            for prompt in prompts:
                SYSTEM_PROMPT = "Please reason step by step, and must put your final answer within \\boxed{}"
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
            
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=1000,truncation=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=8192,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.2
                )

            input_ids_len = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_ids_len:].cpu()
            generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # 3.put results into queue
            for idx, answer in enumerate(generated_answers):
                result_item = batch[idx].copy()
                result_item['generated_answer'] = answer.strip()
                result_queue.put(result_item)
                
    except Exception as e:
        print(f"[Worker-{gpu_id}] An error occurred: {e}")
    finally:
        print(f"[Worker-{gpu_id}] Task completed. (Other workers will complete their tasks silently.)")


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: No results detected GPU。")
        return
        
    all_data = load_all_data(DATA_DIR)
    total_items = len(all_data)
    
    data_chunks = [chunk.tolist() for chunk in np.array_split(all_data, num_gpus)]
    
    for i, chunk in enumerate(data_chunks):
        print(f"GPU {i} will handle {len(chunk)} piece of data。")

    result_queue = mp.Queue()
    processes = []

    print("\n[*] Starting an inference process for each GPU...")
    start_time = time.time()
    
    for i in range(num_gpus):
        process = mp.Process(target=inference_worker, args=(i, data_chunks[i], MODEL_NAME, result_queue))
        processes.append(process)
        process.start()

    final_results = []
    
    with tqdm(total=total_items, desc="overall progress") as pbar:
        while len(final_results) < total_items:
            result = result_queue.get()
            final_results.append(result)
            pbar.update(1)

    print("\n[*] All inference tasks have been completed. Cleaning up child processes...")

    for process in processes:
        process.join()
        
    final_results.sort(key=lambda x: x.get('index', 0))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    end_time = time.time()
    print(f"\n[*] All completed! (Total processing time: [number]) {len(final_results)} Result and save to '{OUTPUT_FILE}'。")
    print(f"[*] Total time spent: {end_time - start_time:.2f} Second。")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
