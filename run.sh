#!/bin/bash

# base LLM+SLM
CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM.py \
 --leader_style=""\
 --test_file="./data/CoGen/test.json" \
 --model_path="./LLaMa-3.3-70B" \
 --output_dir="./result/l+s+base/" \
 --lora_path=""\
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM.py \
 --dir_path="./result/l+s+base/"  \
 --model_path="./LLaMa-3.2-1B" \
 --file_name="LLM_guide.json" \
 --output="slm_response.json" \
 --batch_size=128

python -u 4GPT_eval_relavance.py --file_name='./result/l+s+base/slm_response.json'
python -u 4GPT_eval_personal.py --file_name='./result/l+s+base/slm_response.json'

# base SMFB DPO

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM.py \
 --leader_style="" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/two_tem_dpo/" \
 --lora_path="" \
 --output="LLM_guide_t0.json" \
 --batch_size=32


CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM.py \
 --leader_style="" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=2 \
 --output_dir="./result/two_tem_dpo/" \
 --lora_path="" \
 --output="LLM_guide_t0.5.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM.py \
 --dir_path="./result/two_tem_dpo/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response_t0.json" \
 --file_name="LLM_guide_t0.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM.py \
 --dir_path="./result/two_tem_dpo/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response_t0.5.json" \
 --file_name="LLM_guide_t0.5.json" \
 --batch_size=128

python -u 4GPT_eval_relavance.py --file_name='./result/two_tem_dpo/slm_response_t0.json'
python -u 4GPT_eval_relavance.py --file_name='./result/two_tem_dpo/slm_response_t0.5.json'
python -u 5dpo_sample_tem.py

##################################################
# Here we use llamafactory for fintuning the LLM #
##################################################

# base U-U-RAG

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Directive_Leadership" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader1_train/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Supportive_Leadership" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader2_train/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Participative_Leadership" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader3_train/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Achievement_Oriented_Leadership" \
 --test_file="./data/CoGen/train.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader4_train/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader1_train/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader2_train/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader3_train/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader4_train/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

python -u 4GPT_eval_relavance.py --file_name='./result/leader1_train/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader2_train/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader3_train/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader4_train/slm_response.json'

python -u 8NSGAII_RAG_Leader.py
python -u 8gen_rag_embd.py


CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Directive_Leadership" \
 --test_file="./data/CoGen/test.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader1_test/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Supportive_Leadership" \
 --test_file="./data/CoGen/test.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader2_test/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Participative_Leadership" \
 --test_file="./data/CoGen/test.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader3_test/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0,1,2,3" python -u 1qwen4baseline_LLM_new.py \
 --leader_style="Achievement_Oriented_Leadership" \
 --test_file="./data/CoGen/test.json" \
 --model_path="./LLaMa-3.3-70B" \
 --n=1 \
 --output_dir="./result/leader4_test/" \
 --lora_path="" \
 --output="LLM_guide.json" \
 --batch_size=32

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader1_test/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader2_test/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader3_test/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

CUDA_VISIBLE_DEVICES="0" python -u 2qwen4baseline_SLM_new.py \
 --dir_path="./result/leader4_test/"  \
 --model_path="./LLaMa-3.2-1B" \
 --output="slm_response.json" \
 --file_name="LLM_guide.json" \
 --batch_size=128

python -u 4GPT_eval_relavance.py --file_name='./result/leader1_test/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader2_test/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader3_test/slm_response.json'
python -u 4GPT_eval_relavance.py --file_name='./result/leader4_test/slm_response.json'
python -u 4GPT_eval_personal.py --file_name='./result/leader1_test/slm_response.json'
python -u 4GPT_eval_personal.py --file_name='./result/leader2_test/slm_response.json'
python -u 4GPT_eval_personal.py --file_name='./result/leader3_test/slm_response.json'
python -u 4GPT_eval_personal.py --file_name='./result/leader4_test/slm_response.json'

python -u 8RAG_result.py