import srsly
import numpy as np
from collections import Counter


def calculate_user_data_reference_rate(user_data, generated_output):
    # Exact Match Rate 引出的
    user_words = set(user_data.split())
    output_words = set(generated_output.split())
    matches = user_words.intersection(output_words)
    return len(matches) / len(user_words) if output_words else 0

def cal_UDRR_result_my(data):
    res = []
    for item in data:
        user_data = item["additional_profile"].replace("\n"," ")
        generated_output = item["device_model_output"].replace("\n"," ")
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


def choose_leader_style_rag_privacy(rag_ls):
    idx = [item['leader_style'] for item in rag_ls]
    counter = Counter(idx)
    most_common_element, count = counter.most_common(1)[0]
    # print(most_common_element)
    return most_common_element


def cal_gpt(item):
    try:
        rating  = item.split("Score")[1].split('\n')[0].replace(" ","").replace(":","")
        rating = int(rating)
    except:
        rating = 0
        pass
    return rating

file_name = "./result/leader1_test/slm_response_rela.json"
save_path1 = file_name.replace(".json","_person.json")
data1 = srsly.read_json(save_path1)


file_name = "./result/leader2_test/slm_responserela.json"
save_path2 = file_name.replace(".json","_person.json")
data2 = srsly.read_json(save_path2)


file_name = "./result/leader3_test/slm_response_rela.json"
save_path3 = file_name.replace(".json","_person.json")
data3 = srsly.read_json(save_path3)


file_name = "./result/leader4_test/slm_response_rela.json"
save_path4 = file_name.replace(".json","_person.json")
data4 = srsly.read_json(save_path4)



test_file_name = "./result/rag_idx/test_with_rag.json"


leader_style_all_test = [data1,data2,data3,data4]

data_test = srsly.read_json(test_file_name)

def eval_RAG(data_test,RAG_num):
    gpt_eval_ls = []
    UDRR_score_ls = []
    ppl_ls = []
    for idx,sample in enumerate(data_test):
        reg_res = sample['rag_result']
        leader_style = choose_leader_style_rag_privacy(reg_res[:RAG_num])
        sample['leader_style'] = leader_style
        # print(leader_style)
        related_leader_result = leader_style_all_test[leader_style][idx]
        sample['gpt_eval'] = related_leader_result['gpt_eval']
        sample['gpt_eval_person'] = related_leader_result['gpt_eval_person']
        sample['ppl'] = related_leader_result['ppl']
        gpt_eval_ls.append(cal_gpt(related_leader_result['gpt_eval']))
        UDRR_score_ls.append(cal_gpt(related_leader_result['gpt_eval_person']))
        ppl_ls.append(related_leader_result['ppl'])
    print("UDRR:",np.mean(UDRR_score_ls),np.std(UDRR_score_ls))
    print("GPT:",gpt_res(data_test))
    print('PPL:',ppl_res(data_test))

for rag_num in [1,3,5,10,20,25,30,50]:
    print(rag_num)
    eval_RAG(data_test,rag_num)

