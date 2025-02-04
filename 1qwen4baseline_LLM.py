import pandas as pd
import numpy as np
# from fastchat.conversation import Conversation, SeparatorStyle
from transformers import AutoTokenizer
import os
import sys
import srsly
import fire
from tqdm import tqdm
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_generate,my_format_data_with_server_LLM_leader_style


def main(
        leader_style,
        test_file,
        model_path = "./LLaMa-3.3-70B",
        n=1,
        output_dir=None,
        lora_path = '',
        output=None, 
        batch_size=16):


    print("test_file: ", test_file)
    data = srsly.read_json(test_file)

    model = vllm.LLM(
                model_path,
                worker_use_ray=True,
                tensor_parallel_size=4, 
                gpu_memory_utilization=0.95, 
                trust_remote_code=True,
                dtype="half", 
                enforce_eager=True,
                max_model_len = 2048,
                enable_lora=True,
            )
    tokenizer = model.get_tokenizer()
    is_lora = True if lora_path != '' else False
    data_type = "rec" if 'rec' in test_file else 'write'
    format_input = my_format_data_with_server_LLM_leader_style(data, leader_style,is_lora,data_type)
    model_output_ls = []
    for i in range(0,len(format_input),batch_size):
        format_prompt = format_input[i:i+batch_size]
        model_out,ppl_res = model_generate(model,tokenizer,format_prompt,n=n,lora_path = lora_path)
        model_output_ls+=model_out
    for idx,sample in enumerate(data):
        sample["server_model_output"] = model_output_ls[idx]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(model_path if output_dir is None else output_dir, output if output else "LLM_guide.json")

    srsly.write_json(save_path, data)


if __name__ == "__main__":
    fire.Fire(main)