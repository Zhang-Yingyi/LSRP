import srsly
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
# 定义多目标优化问题

out_dir = "./two_tem_dpo/"

file_name = "./result/two_tem_dpo/slm_response_t0.json"
file_name = file_name.replace(".json","_rela.json")
datat0 = srsly.read_json(file_name)

file_name = "./result/two_tem_dpo/slm_response_t0.5.json"
file_name = file_name.replace(".json","_rela.json")
datat05 = srsly.read_json(file_name)

data_type='rec' if 'rec' in out_dir else 'write'

def calculate_user_data_reference_rate(user_data, generated_output):
    # Exact Match Rate 引出的
    user_words = set(user_data.split())
    output_words = set(generated_output.split())
    matches = user_words.intersection(output_words)
    return len(matches) / len(user_words) if output_words else 0

def cal_UDRR_sample(item):
    user_data = item["additional_profile"].replace("\n"," ")
    generated_output = item["device_model_output"].replace("\n"," ")
    score = calculate_user_data_reference_rate(user_data, generated_output)
    return score
def cal_GPT_score(item):
    try:
        rating  = item['gpt_eval'].split("**Rating:**")[1].replace(" ","").replace("\n","")[0]
        res = int(rating)
    except:
        res = 0
    return res

overall_matrix_tem = [] #[n*4*3]
for sample in zip(datat0,datat05):
    task = sample[0]['conversations'][0]['value']
    privacy = sample[0]['additional_profile']
    value = [] #[4*3]
    for item in sample:
        temp_score = [cal_UDRR_sample(item),item['ppl'],cal_GPT_score(item)]
        value.append(temp_score)
    overall_matrix_tem.append(value)

np_overall_matrix_tem = np.array(overall_matrix_tem)
# 这里用的正太分布归一化有待商榷

normalized_matrix_tem = np.zeros_like(np_overall_matrix_tem)
for i in range(np_overall_matrix_tem.shape[2]):
    feature = np_overall_matrix_tem[:, :, i]
    mean_val = np.mean(feature)
    std_val = np.std(feature)
    if std_val == 0:  # 避免分母为零
        normalized_matrix_tem[:, :, i] = 0
    else:
        normalized_matrix_tem[:, :, i] = (feature - mean_val) / std_val


class MultiObjectiveProblem(Problem):
    def __init__(self,np_overall_matrix,data_matrix,n_var=3, n_obj=3, n_constr=0, xl=-10,xu=10):
        super().__init__(n_var=n_var,  # 决策变量数
                         n_obj=n_obj,  # 目标函数数
                         n_constr=n_constr,  # 无约束条件
                         xl=xl,  # 决策变量下界
                         xu=xu)   # 决策变量上界
        self.np_overall_matrix = np_overall_matrix
        self.data_matrix = data_matrix

    def _evaluate(self, x, out, *args, **kwargs):
        # 定义三个目标函数
        res = self.porcess_result_pop(x) #[100,3]
        f1 = -res[:,0]  # 目标函数1
        f2 = -res[:,1]  # 目标函数2
        f3 = -res[:,2]  # 目标函数3

        out["F"] = np.column_stack([f1, f2, f3])

    def porcess_result_pop(self,x):
        res_all = []
        for idx in range(x.shape[0]):
            res = self.process_result(x[idx,:])
            res_all.append(res)
        return np.array(res_all) # [100,3]
    
    def process_result(self, x):
        result = self.data_matrix@x.T
        idx = np.argmax(result, axis=1)
        result = self.np_overall_matrix[np.arange(self.np_overall_matrix.shape[0]), idx, :]
        return np.mean(result, axis = 0) #[3]

problem = MultiObjectiveProblem(np_overall_matrix_tem,normalized_matrix_tem)

# 使用 NSGA-II 算法并结合 CMA 风格的采样和变异
algorithm = NSGA2(
    pop_size=100,
    sampling=LHS(),  # 拉丁超立方体采样
    crossover=SBX(prob=0.9, eta=15),  # 二次边界交叉
    mutation=GaussianMutation(prob=0.2, sigma=0.1),  # 高斯变异
)

# 定义终止条件
termination = get_termination("n_gen", 100)

# 执行优化
result = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

# 可视化结果
plot = Scatter()
plot.add(result.F, facecolor="red", edgecolor="black")
plot.show()

# 输出 Pareto 前沿结果
print("Pareto solutions:\n", result.X)
print("Pareto front:\n", result.F)



F = result.F

print("Three objects is ")
print(
    F[np.argmax(-F[:,0]),:],'\n',
    F[np.argmax(-F[:,1]),:],'\n',
    F[np.argmax(-F[:,2]),:],'\n',
)

print(
    np.exp(F[np.argmax(-F[:,0]),1]),'\n',
    np.exp(F[np.argmax(-F[:,1]),1]),'\n',
    np.exp(F[np.argmax(-F[:,2]),1]),'\n',
)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(F):
    """对每个解的目标向量使用 L2 范数归一化，使每个解的目标向量长度为1。"""
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-16
    return F / norms

def find_knee_point(X, F):
    """
    寻找拐点示例（已修改以同时返回对应的X）。
    输入：
        X: (N, n_var) 决策变量矩阵
        F: (N, 3)    目标值矩阵
    输出：
        knee_sol_F: 拐点对应的目标值向量
        knee_sol_X: 拐点对应的决策变量向量
        knee_idx:    拐点在排序序列中的索引（用于调试或扩展）
        idx:         排序后的索引映射
    """
    F_norm = normalize(F)
    # 根据第一个目标进行排序
    idx = np.argsort(F_norm[:,0])
    F_sorted = F_norm[idx]

    angles = []
    for i in range(1, len(F_sorted)-1):
        A = F_sorted[i-1]
        B = F_sorted[i]
        C = F_sorted[i+1]

        AB = B - A
        BC = C - B

        dot_product = np.dot(AB, BC)
        norm_AB = np.linalg.norm(AB)
        norm_BC = np.linalg.norm(BC)
        cos_theta = dot_product / (norm_AB * norm_BC + 1e-16)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        angles.append(angle)

    # 找到最大弯曲度所在的位置
    knee_idx = np.argmax(angles) + 1  # 对应F_sorted中的位置
    # 从原始F、X中提取拐点对应解
    global_knee_idx = idx[knee_idx]
    knee_sol_F = F[global_knee_idx]
    knee_sol_X = X[global_knee_idx]

    return knee_sol_F, knee_sol_X, knee_idx, idx


# result.X 为决策变量矩阵 (N, n_var)
# result.F 为目标矩阵 (N, 3)
X = result.X
F_original = result.F.copy()  # 保留原F
F_demo = F_original.copy()   # 用于演示的副本

# 对 F_demo 进行排序仅用于展示，这里根据需要，你也可不对其排序
F_demo = np.sort(F_demo, axis=0)

knee_sol_F, knee_sol_X, k_idx, sorted_idx = find_knee_point(X, F_original)
print("Knee Point (original scale) F:", knee_sol_F)
print("Knee Point (original scale) X:", knee_sol_X)


################## 拐点选择最优训练策略 ##################

result_knee = normalized_matrix_tem @ knee_sol_X.T
match_idx = np.argmax(result_knee, axis=1)
match_idx = match_idx.tolist()

sample_accept_res = []
sample_reject_res = []
sample1_win_num = 0
sample2_win_num = 0
for idx,sample in enumerate(zip(datat0,datat05)):
    if match_idx[idx]==0:
        sample1_win_num+=1
        sample_accept_res.append(sample[0])
        sample_reject_res.append(sample[1])
    elif match_idx[idx]==1:
        sample2_win_num+=1
        sample_accept_res.append(sample[1])
        sample_reject_res.append(sample[0])
    else:
        continue
print(len(sample_accept_res),len(sample_reject_res))
print(sample1_win_num,sample2_win_num)

from utils import my_format_data_with_server_LLM,chossen_template_LLM,my_format_data_with_server_dpo

format_input = my_format_data_with_server_dpo(sample_accept_res)
format_prompt_accpet = chossen_template_LLM(sample_accept_res)
format_prompt_reject = chossen_template_LLM(sample_reject_res)
dpo_train_sample = []
for question,accept,reject in zip(format_input,format_prompt_accpet,format_prompt_reject):

    dpo_template =   {
        "conversations": [
        {
            "from": "human",
            "value": question[1]["content"]
        }
        ],
        "chosen": {
        "from": "gpt",
        "value": accept
        },
        "rejected": {
        "from": "gpt",
        "value": reject
        }
    }
    dpo_train_sample.append(dpo_template)
srsly.write_json(out_dir+"dpo_train_sample_knee.json", dpo_train_sample)
