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

from utils import format_data_with_device_SLM,model_generate,evaluate_model,my_format_data_with_device_SLM




def main(dir_path,
        model_path = "./LLaMa-3.2-1B",
        output=None, 
        file_name = "LLM_guide.json",
        batch_size=16):
    test_file = dir_path+file_name
    print("test_file: ", test_file)
    output_dir = dir_path

    data = srsly.read_json(test_file)

    data = srsly.read_json(test_file)

    model = vllm.LLM(
                model_path,
                worker_use_ray=True,
                tensor_parallel_size=1, 
                gpu_memory_utilization=0.9, 
                trust_remote_code=True,
                dtype="half", 
                enforce_eager=True,
                max_model_len = 4096,
                enable_lora=True,
            )
    tokenizer = model.get_tokenizer()

    bleu_score_ls=[]
    rouge_1_ls=[]
    rouge_2_ls=[]
    rouge_l_ls=[]
    data_type = "rec" if 'rec' in test_file else 'write'
    format_input = my_format_data_with_device_SLM(data,data_type)
    model_output_ls = []
    ppl_res_all = []
    for i in range(0,len(format_input),batch_size):
        format_prompt = format_input[i:i+batch_size]
        model_out,ppl_res = model_generate(model,tokenizer,format_prompt,max_tokens=768,ppl=True)
        model_output_ls+=model_out
        ppl_res_all+=ppl_res
    # print(ppl_res_all)
        

    for idx,sample in enumerate(data):
        sample["device_model_output"] = model_output_ls[idx]
        sample["ppl"] = ppl_res_all[idx]

        
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(model_path if output_dir is None else output_dir, output if output else "slm_response.json")

    srsly.write_json(save_path, data)


if __name__ == "__main__":
    fire.Fire(main)