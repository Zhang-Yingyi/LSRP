import asyncio
import aiohttp
import srsly
import http.client
import json
import fire
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import srsly
from tqdm import tqdm

def calculate_user_data_reference_rate(user_data, generated_output):
    # Exact Match Rate 引出的
    user_words = set(user_data.split())
    output_words = set(generated_output.split())
    matches = user_words.intersection(output_words)
    return len(matches) / len(user_words) if output_words else 0

def cal_UDRR_result_my(data):
    res = []
    for item in tqdm(data):
        user_data = item["additional_profile"].replace("\n"," ")
        generated_output = item["device_model_output"].replace("\n"," ")
        score = calculate_user_data_reference_rate(user_data, generated_output)
        res.append(score)
    return np.mean(np.array(res))

def cal_UDRR_result(data):
    res = []
    for item in tqdm(data):
        user_data = item["additional_profile"].replace("\n"," ")
        generated_output = item["model_output"].replace("\n"," ")
        score = calculate_user_data_reference_rate(user_data, generated_output)
        res.append(score)
    return np.mean(np.array(res))


def gpt_res(data):
    rating_ls = []
    error = 0
    for item in data:
        try:
            rating  = item['gpt_eval'].split("Score")[1].split('\n')[0].replace(" ","").replace(":","")
            rating_ls.append(int(rating))
        except:
            error+=1
            pass
    return np.mean(rating_ls),np.std(rating_ls),error

def ppl_res(data):
    ppl_ls = []
    error = 0
    for item in data:
        ppl = item['ppl']
        ppl_ls.append(ppl)
    return np.exp(-np.mean(ppl_ls))

API_KEY = 'sk-8WFPcdwKoCURDmFR5c4c13EdD486435cB15f0bAcB247470b'

BASE_URL = "https://api.gpt.ge/v1/"
# BASE_URL = "https://run.v36.cm/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}



eval_genre_prompt = """You are given a question, a user profile, and an answer. Evaluate how well the answer matches the user’s demand and its relevance to the question. Assign an integer score from 0 to 10, where:
	•	0 indicates the answer is irrelevant.
	•	10 indicates the answer is totally relevant.

Question: {question}
User Profile: {profile}
Answer: {response}

Provide only the score as the final output in the following format:
Relevance Score: X
"""
# LLaMa

def main_eval(file_name):
    # file_name = "./result/l+s+base/slm_response.json"

    save_path = file_name.replace(".json","_rela.json")
    data = srsly.read_json(file_name)

    prompts = []
    for item in tqdm(data):
        user_input = eval_genre_prompt.format(
            question=item["conversations"][0]["value"],
            profile = item['additional_profile'],
            response=item["device_model_output"]
        )
        prompts.append(user_input)


    async def create_completion(session, content):
        try:
            async with session.post(
                url=f"{BASE_URL}chat/completions",
                json={
                    "model": "gpt-4o",
                    "max_tokens": 30,
                    "temperature": 0.0000000000000000000000,
                    "messages": [{"role": "user", "content": content[1]}],
                },
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    completion = result['choices'][0]['message']['content']
                    content[0]['gpt_eval'] = completion
                    return completion
                else:
                    print(f"请求失败，状态码: {response.status}")
                    print(f"响应内容: {await response.text()}")
                    content[0]['gpt_eval'] = ''
                    return None
        except Exception as e:
            print(f"请求发生异常: {e}")
            return None
        
    async def main():
        max_limits = 500  # 设置一个合理的并发请求数
        semaphore = asyncio.Semaphore(max_limits)
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []

            for content in zip(data,prompts):
                task = asyncio.create_task(limited_create_completion(session, content, semaphore))
                tasks.append(task)

            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                result = await future
                results.append(result)

        return results

    async def limited_create_completion(session, content, semaphore):
        async with semaphore:
            return await create_completion(session, content)
        

    responses = asyncio.run(main())
    UDRR_res = []
    # 将结果保存到文件或与原数据合并
    for i, item in enumerate(data):
        # item['gpt_eval'] = responses[i] if responses[i] is not None else "请求失败或无响应"
        user_data = item["additional_profile"].replace("\n"," ")
        generated_output = item["device_model_output"].replace("\n"," ")
        score = calculate_user_data_reference_rate(user_data, generated_output)
        item['UDRR_score'] = score
        UDRR_res.append(score)

    # 保存到新的 JSON 文件
    srsly.write_json(save_path, data)
    print(f"结果已保存到 {save_path}")



    print("UDRR:",np.mean(UDRR_res))
    print("GPT:",gpt_res(data))
    print('PPL:',ppl_res(data))

if __name__ == "__main__":
    fire.Fire(main_eval)