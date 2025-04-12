import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.selfrepresentation import SSCOMPOD
from src.DFNO import DFNO
from sklearn.metrics import precision_score, recall_score, f1_score
from main import SCNFOD


def analyze_results(file_path):
    """分析结果文件，查找SCNFOD值同时大于SSCOMPOD和DFNO的情况"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"正在分析文件: {file_path}")
    results = []
    current_params = None
    current_scores = {}

    # 解析文件内容
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 遇到新的实验组时，保存前一组数据
        if line.startswith("====="):
            if current_params and len(current_scores) >= 3:
                results.append((current_params, current_scores.copy()))
            current_params = None
            current_scores = {}
            continue

        # 读取参数信息
        if line.startswith("Parameters:"):
            current_params = line.replace("Parameters:", "").strip()
            continue

        # 读取评分数据
        if ":" in line and not line.startswith("====="):
            method, value = line.split(":", 1)
            method = method.strip()
            try:
                value = float(value.strip())
                current_scores[method] = value
            except ValueError:
                continue

    # 添加最后一组数据
    if current_params and len(current_scores) >= 3:
        results.append((current_params, current_scores.copy()))

    # 筛选并打印符合条件的结果
    found = False
    for params, scores in results:
        # 找出SCNFOD开头的键
        scnfod_keys = [k for k in scores.keys() if k.startswith("SCNFOD")]
        if scnfod_keys and "SSCOMPOD" in scores and "DFNO" in scores:
            scnfod_key = scnfod_keys[0]
            scnfod = scores[scnfod_key]
            sscompod = scores["SSCOMPOD"]
            dfno = scores["DFNO"]

            # 检查SCNFOD是否同时大于SSCOMPOD和DFNO
            if scnfod > sscompod and scnfod > dfno:
                found = True
                print(f"Parameters: {params}")
                print(f"SSCOMPOD: {sscompod:.4f}")
                print(f"DFNO: {dfno:.4f}")
                print(f"{scnfod_key}: {scnfod:.4f}")
                print("-" * 40)

    if not found:
        print("没有找到SCNFOD值同时大于SSCOMPOD和DFNO的情况")


def find_best_results(file_path):
    """分析结果文件，找出SCNFOD方法相比SSCOMPOD和DFNO增益最大的参数配置"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"正在分析文件: {file_path}")

    results = []
    current_params = None
    current_scores = {}

    # 解析文件内容
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 遇到新的实验组时，保存前一组数据
        if line.startswith("====="):
            if current_params and len(current_scores) >= 3:
                results.append((current_params, current_scores.copy()))
            current_params = None
            current_scores = {}
            continue

        # 读取参数信息
        if line.startswith("Parameters:"):
            current_params = line.replace("Parameters:", "").strip()
            continue

        # 读取评分数据
        if ":" in line and not line.startswith("====="):
            method, value = line.split(":", 1)
            method = method.strip()
            try:
                value = float(value.strip())
                current_scores[method] = value
            except ValueError:
                continue

    # 添加最后一组数据
    if current_params and len(current_scores) >= 3:
        results.append((current_params, current_scores.copy()))

    # 计算SCNFOD相比SSCOMPOD和DFNO的增益，找出总增益最大的参数配置
    best_gain = float("-inf")
    best_params = None
    best_scores = None

    for params, scores in results:
        # 找出SCNFOD开头的键
        scnfod_keys = [k for k in scores.keys() if k.startswith("SCNFOD")]
        if scnfod_keys and "SSCOMPOD" in scores and "DFNO" in scores:
            scnfod_key = scnfod_keys[0]
            scnfod = scores[scnfod_key]
            sscompod = scores["SSCOMPOD"]
            dfno = scores["DFNO"]

            # 计算增益
            gain_over_sscompod = scnfod - sscompod
            gain_over_dfno = scnfod - dfno
            total_gain = gain_over_sscompod + gain_over_dfno

            # 更新最佳结果
            if total_gain > best_gain:
                best_gain = total_gain
                best_params = params
                best_scores = {
                    "SCNFOD": scnfod,
                    "SSCOMPOD": sscompod,
                    "DFNO": dfno,
                    "gain_over_sscompod": gain_over_sscompod,
                    "gain_over_dfno": gain_over_dfno,
                    "total_gain": total_gain,
                    "scnfod_key": scnfod_key,
                }

    # 打印最佳结果
    if best_params:
        print(f"最佳参数配置: {best_params}")
        print(f"AUC值:")
        print(f"  SSCOMPOD: {best_scores['SSCOMPOD']:.4f}")
        print(f"  DFNO: {best_scores['DFNO']:.4f}")
        print(f"  {best_scores['scnfod_key']}: {best_scores['SCNFOD']:.4f}")
        print(f"相比SSCOMPOD的增益: {best_scores['gain_over_sscompod']:.4f}")
        print(f"相比DFNO的增益: {best_scores['gain_over_dfno']:.4f}")
        print(f"总增益: {best_scores['total_gain']:.4f}")
    else:
        print("未找到有效的比较结果")
    print("-" * 40)


def anlayze_results():
    import os

    file_path_list = [
        os.path.join("results", f) for f in os.listdir("results") if f.endswith(".txt")
    ]
    for file_path in file_path_list:
        analyze_results(file_path)
    for file_path in file_path_list:
        find_best_results(file_path)


def _detect_anomalies(scores, threshold):
    anomaly_indices = np.where(scores >= threshold)[0]
    return anomaly_indices


def anomaly_detection(
    k,
    Alpha,
    data_filepath,
    threshold=0.5,
    data_threshold=7000,
    is_similarity_matrix=False,
):
    # Not used function `anomaly_detection()`
    data = pd.read_csv(data_filepath)
    if len(data) > data_threshold:
        print(f"数据集大小为 {len(data)} 条记录，随机抽取 {data_threshold} 条")
        data = data.sample(n=data_threshold, random_state=42)
    if (
        "dataset\\CIC-IDS-2017\\MachineLearningCVE" in data_filepath
        or "dataset/CIC-IDS-2017/MachineLearningCVE" in data_filepath
    ):
        # 假设最后一列是标签列
        last_column = data.columns[-1]
        print(f"处理CIC-IDS-2017数据集，将最后一列'{last_column}'编码为二分类")

        # 检查类型并转换为字符串
        if data[last_column].dtype != "object":
            data[last_column] = data[last_column].astype(str)

        # 将包含"BENIGN"的记录编码为0，其他编码为1
        data[last_column] = data[last_column].apply(lambda x: 0 if "BENIGN" in x else 1)

        # 显示标签分布
        benign_count = sum(data[last_column] == 0)
        attack_count = sum(data[last_column] == 1)
        print(f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条")

    if (
        "dataset\\kddcup.data" in data_filepath
        or "dataset/kddcup.data" in data_filepath
    ):
        # 假设最后一列是标签列
        last_column = data.columns[-1]
        print(f"处理kddcup数据集，将最后一列'{last_column}'编码为二分类")

        # 检查类型并转换为字符串
        if data[last_column].dtype != "object":
            data[last_column] = data[last_column].astype(str)

        # 将包含"BENIGN"的记录编码为0，其他编码为1
        data[last_column] = data[last_column].apply(
            lambda x: 0 if "normal." in x else 1
        )

        # 显示标签分布
        benign_count = sum(data[last_column] == 0)
        attack_count = sum(data[last_column] == 1)
        print(f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条")
    if "dataset\\UNSW-NB15" in data_filepath or "dataset/UNSW-NB15" in data_filepath:
        # 获取标签列
        last_column = data.columns[-1]
        print("UNSW-NB15数据集，0 for normal and 1 for attack records, 无需处理")

        # 确保标签列为字符串类型
        data[last_column] = data[last_column].astype(str)

        # 使用向量化操作计算标签分布
        benign_count = sum(data[last_column] == "0")
        attack_count = sum(data[last_column] == "1")
        print(f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条")

    label_encoders = {}
    for column in data.columns:
        if data[column].dtype != "int64" and data[column].dtype != "float64":
            print(f"列号: {column}")
            data[column] = data[column].astype(str)
            label_encoders[column] = LabelEncoder()
            data[column] = label_encoders[column].fit_transform(data[column])

            encoding_map = dict(
                zip(
                    label_encoders[column].classes_,
                    label_encoders[column].transform(label_encoders[column].classes_),
                )
            )
    tran_data = data.values
    tran_data = np.nan_to_num(tran_data, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler()
    tran_data_scalSSC = scaler.fit_transform(tran_data)
    tran_data_scalDFNO = scaler.fit_transform(tran_data)
    y_true = tran_data[:, -1]
    subspaceAnomalyScore, affinity_matrix = SSCOMPOD(data=tran_data_scalSSC)
    DFNOAnomalyScore = DFNO(
        data=tran_data_scalDFNO,
        affinity_matrix=affinity_matrix,
        k=k,
        is_similarity_matrix=is_similarity_matrix,
    )
    weightAnomalyScore = SCNFOD(
        subspaceAnomalyScore=subspaceAnomalyScore,
        DFNOAnomalyScore=DFNOAnomalyScore,
        Alpha=Alpha,
    )

    scores_dict = {
        "SSCOMPOD": subspaceAnomalyScore,
        "DFNO": DFNOAnomalyScore,
        "SCNFOD(α={:.1f})".format(Alpha): weightAnomalyScore,
    }

    anomalies_dict = {}
    for method_name, scores in scores_dict.items():
        anomaly_indices = _detect_anomalies(scores, threshold)
        y_pred = np.zeros_like(y_true)
        y_pred[anomaly_indices] = 1  # 将检测到的异常点标记为1

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{method_name}检测结果:")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall): {recall:.4f}")
        print(f"  F1值: {f1:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    analyze_results()
