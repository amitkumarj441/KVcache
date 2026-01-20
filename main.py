import os
import json
import torch
import numpy as np
import multiprocessing as mp
from transformers import AutoTokenizer, Qwen3ForCausalLM
from tqdm import tqdm
import time
from queue import Empty

from baselines import init_kvc

MODEL_NAME = "/amit/hf_download/models/Qwen3-4B"
INPUT_FILE = "data/qwen_8k_2k/baseline.jsonl"
OUTPUT_FILE = "data/qwen_8k_2k/kvc.jsonl"
Cache = init_kvc()

WORKER_BATCH_SIZE = 16
# Token length threshold for conditional reasoning
TOKEN_THRESHOLD = 2048

def load_and_filter_data(filepath: str, tokenizer: AutoTokenizer, threshold: int):
    """
    Load data from the specified file and filter it based on the length of the 'generated_answer' token。
    """
    to_process = []
    to_copy = []
    print(f"[*] loading '{filepath}' ...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="process..."):
            item = json.loads(line)
            answer_text = item.get('generated_answer', '')
            
            token_ids = tokenizer(answer_text, return_tensors='pt').input_ids
            
            if token_ids.shape[1] < threshold:
                to_copy.append(item)
            else:
                to_process.append(item)

    print(f"[*] Screening complete. Further reasoning is required: {len(to_process)} Item, copy directly: {len(to_copy)} 。")
    return to_process, to_copy

def inference_worker(gpu_id: int, data_chunk: list, model_name: str, result_queue: mp.Queue):
    """
    Run the inference subprocess function on the specified GPU using our KV cache.
    """
    device = f"cuda:{gpu_id}"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager" # flash_attention_2
        ).to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        for i in range(0, len(data_chunk), WORKER_BATCH_SIZE):
            # 2. [Core Changes] Reset/Reinitialize for each batch cache
            past_key_values = Cache(cache_budget=2048)
            
            batch = data_chunk[i:i + WORKER_BATCH_SIZE]
            
            # Note: Here we use the 'question' field to generate new answers.
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
                    max_new_tokens=8192, # Adjust as needed
                    do_sample=False,
                    past_key_values=past_key_values,
                    use_cache=True,
                    repetition_penalty=1.2
                )

            input_ids_len = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_ids_len:].cpu()
            generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for idx, answer in enumerate(generated_answers):
                result_item = batch[idx].copy()
                result_item['generated_answer'] = answer.strip()
                result_queue.put(result_item)
                
    except Exception as e:
        print(f"[Worker-{gpu_id}] An error occurred: {e}")
    finally:
        print(f"[Worker-{gpu_id}] Task completed。")
        result_queue.put(None) 


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: No GPU detected。")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    to_process, to_copy = load_and_filter_data(INPUT_FILE, tokenizer, TOKEN_THRESHOLD)
    
    if not to_process:
        print("[*] No data requires re-inference. Save the results directly.。")
        final_results = sorted(to_copy, key=lambda x: x.get('index', 0))
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in final_results:
                f.write(json.dumps(item) + '\n')
        print(f"[*] All done! Results have been saved. '{OUTPUT_FILE}'。")
        return

    total_items_to_process = len(to_process)
    data_chunks = [chunk.tolist() for chunk in np.array_split(to_process, num_gpus)]
    
    for i, chunk in enumerate(data_chunks):
        print(f"GPU {i} will handle {len(chunk)} piece of data。")

    result_queue = mp.Queue()
    processes = []

    print("\n[*] Initiating our inference process for each GPU...")
    start_time = time.time()
    
    for i in range(num_gpus):
        process = mp.Process(target=inference_worker, args=(i, data_chunks[i], MODEL_NAME, result_queue))
        processes.append(process)
        process.start()

    processed_results = []
    done_workers = 0 
    
    with tqdm(total=total_items_to_process, desc="Reasoning progress") as pbar:
        while done_workers < num_gpus:
            try:
                result = result_queue.get(timeout=10000) # 60-second timeout
                if result is None:
                    done_workers += 1
                else:
                    processed_results.append(result)
                    pbar.update(1)
            except Empty:
                pbar.write("!!!! Warning: Timeout! A child process may have crashed。 !!!!")
                break 

    pbar.close()
    print(f"\n[*] The reasoning cycle ends。{done_workers}/{num_gpus} Each worker process has sent a completion signal。")

    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            print(f"Warning: process {process.pid} Unable to join normally, will be forcibly terminated.。")
            process.terminate()
    
    final_results = to_copy + processed_results
    print(f"[*] Collected in total {len(final_results)} Overall results. Currently being organized and saved....")
    final_results.sort(key=lambda x: x.get('index', 0))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    end_time = time.time()
    print(f"\n[*] All done! Results have been saved. '{OUTPUT_FILE}'。")
    print(f"[*] Total time spent: {end_time - start_time:.2f} Second。")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
