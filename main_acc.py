import json
from tqdm import tqdm
import re
from latex2sympy2 import latex2sympy
from sympy import simplify
import multiprocessing

baseline_path = "/data/qwen_8k_2k/baseline.jsonl"
num_processes = 32

def extract_boxed_answer(solution_str):
    match_start = re.search(r'\\boxed{', solution_str)
    if not match_start:
        return None
    
    start_index = match_start.end()
    level = 1
    
    # With a slight optimization, if the string is very long, pre-calculate the length.
    str_len = len(solution_str)
    for i in range(start_index, str_len):
        char = solution_str[i]
        if char == '{':
            level += 1
        elif char == '}':
            level -= 1
        
        if level == 0:
            # A matching closing bracket was found.
            return solution_str[start_index:i]
            
    return None # No matching closing bracket found

def normalize_math(s):
    if not s:
        return ''
    s = re.sub(r'\s+|\\,|\\quad|\\qquad|\\!|\\;', '', s)
    s = re.sub(r',', '', s)  # Remove thousands separator commas 
    
    s = re.sub(r'\\(d|t)frac\b', r'\\frac', s)
    s = re.sub(r'\\frac(\d+)(\d+)', r'\\frac{\1}{\2}', s)

    s = re.sub(r'^([A-Z])$', r'\\text{(\1)}', s)
    
    return s.strip()

def adjust_extracted_format(extracted_norm, answer_norm):
    if re.fullmatch(r'\\text\{([^{}]+)\}', answer_norm) and '\\text' not in extracted_norm:
        return f'\\text{{{extracted_norm}}}'
    if answer_norm.startswith('\\$') and '\\$' not in extracted_norm:
        return f'\\${extracted_norm}'
    if '^\\circ' in answer_norm and '^\\circ' not in extracted_norm:
        return f'{extracted_norm}^\\circ'
    return extracted_norm

def is_math_equivalent(expr1, expr2):
    if len(expr1) > 50 or len(expr2) > 50:
        return False
    try:
        expr1_sympy = latex2sympy(expr1)
        expr2_sympy = latex2sympy(expr2)
        
        if simplify(expr1_sympy - expr2_sympy) == 0:
            return True
    except Exception:
        pass
        
    # As a last resort, perform raw string comparison.
    return expr1 == expr2

def evaluate_math(solution, answer):
    extracted = extract_boxed_answer(solution) 
    answer_norm = normalize_math(answer)
    extracted_norm = normalize_math(extracted) if extracted else None
    
    if extracted_norm is None:
        return False
    
    if extracted_norm and answer_norm:
        extracted_norm = adjust_extracted_format(extracted_norm, answer_norm)
        
    if extracted_norm.lower() == answer_norm.lower():
        return True
        
    if is_math_equivalent(extracted_norm, answer_norm):
        return True
    
    return False

def process_line(line, dataset_entries):
    data = json.loads(line)
    match_key = (data["index"], data["question"])
    
    for dataset_name, entries in dataset_entries.items():
        if match_key in entries:
            is_correct = evaluate_math(data["generated_answer"], str(data["true_answer"]))
            return (dataset_name, is_correct)
            
    return (None, False) 


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1.
    dataset_config = {
        "olympiad": "/data/olympiad.jsonl",
        "minerva": "/data/minerva.jsonl",
        "gpqa": "/data/gpqa.jsonl",
        "aime25": "/data/aime25.jsonl",
        "amc": "/data/amc.jsonl",
        "math": "/data/math.jsonl",
        "aime24": "/data/aime24.jsonl",
        "gsm8k": "/data/gsm8k.jsonl"
    }

    # 2.Preprocessing: Read the raw dataset (this part is fast and does not require parallelization)
    print("Preloading dataset metadata...")
    dataset_entries = {}
    for dataset_name, file_path in dataset_config.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries = set()
                for line in f:
                    data = json.loads(line)
                    key = (data.get("index"), data.get("question"))
                    entries.add(key)
                dataset_entries[dataset_name] = entries
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}，This data set will be skipped。")
            dataset_entries[dataset_name] = set()

    # 3. Configure and handle multiple processes. (baseline.jsonl)
    counts = {
        name: {"total": 0, "correct": 0, "accuracy": 0.0}
        for name in dataset_config
    }

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            lines = f.readlines() 
        
        print(f"use {num_processes} each process performs parallel processing....")

        with multiprocessing.Pool(processes=num_processes) as pool:
            from functools import partial
            worker_func = partial(process_line, dataset_entries=dataset_entries)
            
            results = list(tqdm(pool.imap_unordered(worker_func, lines), total=len(lines), desc="Evaluate progress"))

        for dataset_name, is_correct in results:
            if dataset_name and dataset_name in counts:
                counts[dataset_name]["total"] += 1
                if is_correct:
                    counts[dataset_name]["correct"] += 1
    
    except FileNotFoundError:
        print(f"Error: Not found baseline document: {baseline_path}")
        exit()

    # 5. Calculate and output the final result.
    for dataset, stats in counts.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        print(f"{dataset:>8}:", end="\t")
        print(f"  total quantity: {stats['total']:>8}", end="\t")
        print(f"  Correct quantity: {stats['correct']:>8}", end="\t")
        print(f"  Accuracy: {stats['accuracy']:.4f}", end="\t")
        print()
