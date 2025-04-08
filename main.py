import numpy as np
import matplotlib
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io import loadmat
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.selfrepresentation import SSCOMPOD
from src.DFNO import DFNO
from src.draw import calculate_auc_and_save
import tqdm

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


def SCNFOD(subspaceAnomalyScore, DFNOAnomalyScore, Alpha=0.5):
    return Alpha * subspaceAnomalyScore + (1 - Alpha) * DFNOAnomalyScore


def detect_anomalies(scores, threshold):
    """
    使用阈值检测异常点

    参数:
    scores: 异常分数
    threshold: 异常分数阈值，分数大于等于此值的点被判定为异常

    返回:
    anomaly_indices: 异常点的索引列表
    """
    anomaly_indices = np.where(scores >= threshold)[0]
    return anomaly_indices


def preprocess_discrete_variables(data, categorical_cols=None, method="label"):
    """
    将离散变量预处理为连续变量

    参数:
    data: numpy数组或pandas DataFrame，包含要预处理的数据
    categorical_cols: 离散变量的列索引列表，如果为None，则自动检测
    method: 预处理方法，可选 'label'(标签编码) 或 'onehot'(独热编码)

    返回:
    processed_data: 预处理后的数据(numpy数组)
    """
    # 如果输入是numpy数组，转换为pandas DataFrame便于处理
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    # 如果没有指定分类变量列，尝试自动检测
    if categorical_cols is None:
        categorical_cols = []
        for col in df.columns:
            # 检查是否为可能的分类变量：唯一值数量较少且为整数
            unique_values = df[col].nunique()
            if unique_values < 10 and pd.api.types.is_integer_dtype(df[col]):
                categorical_cols.append(col)

    # 应用选定的预处理方法
    if method == "label":
        # 标签编码：将类别变量转换为0到n-1的整数
        for col in categorical_cols:
            le = LabelEncoder()
            # 处理可能的NaN值
            not_null = df[col].notna()
            if not_null.all():
                df[col] = le.fit_transform(df[col])
            else:
                df.loc[not_null, col] = le.fit_transform(df.loc[not_null, col])

        return df.values

    elif method == "onehot":
        # 独热编码：为每个类别创建一个新的二元特征
        # 保留非分类变量
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        numerical_data = df[numerical_cols]

        # 对分类变量进行独热编码
        if categorical_cols:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            categorical_data = df[categorical_cols]
            encoded_data = encoder.fit_transform(categorical_data)

            # 创建独热编码后的DataFrame
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[
                    f"{col}_{cat}"
                    for col, cats in zip(categorical_cols, encoder.categories_)
                    for cat in cats
                ],
            )

            # 合并数值数据和编码后的数据
            if not numerical_data.empty:
                result = pd.concat(
                    [
                        numerical_data.reset_index(drop=True),
                        encoded_df.reset_index(drop=True),
                    ],
                    axis=1,
                )
            else:
                result = encoded_df

            return result.values
        else:
            # 如果没有分类变量，直接返回
            return numerical_data.values

    else:
        raise ValueError(f"不支持的预处理方法: {method}。请使用 'label' 或 'onehot'")


def main(dataset_dir, data_threshold=7000):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    k_list = [2, 11, 20, 29, 38, 47, 56, 65]
    Alpha_list = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9]
    is_similarity_matrix = False
    data_filepath_list = [
        (os.path.join(dataset_dir, filename), filename)
        for filename in os.listdir(dataset_dir)
        if filename.endswith(".csv")
    ]
    for i, (data_filepath, dataset_name) in enumerate(data_filepath_list):
        # if os.path.getsize(data_filepath) > file_threshold * 1024 * 1024:
        #     print(
        #         f"跳过文件: {dataset_name}，大小{os.path.getsize(data_filepath)/(1024 * 1024):.4f}MB超过阈值"
        #     )
        #     continue

        if os.path.exists(f"./results/{dataset_name}.txt"):
            print(f"跳过文件: {dataset_name}，结果文件已存在")
            continue

        print(">" * 50)
        print(f"数据集{i}/{len(data_filepath_list)}: {dataset_name}")
        data = pd.read_csv(data_filepath)

        if len(data) > data_threshold:
            print(f"数据集大小为 {len(data)} 条记录，随机抽取 {data_threshold} 条")
            data = data.sample(n=data_threshold, random_state=42)

        try:
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
                data[last_column] = data[last_column].apply(
                    lambda x: 0 if "BENIGN" in x else 1
                )

                # 显示标签分布
                benign_count = sum(data[last_column] == 0)
                attack_count = sum(data[last_column] == 1)
                print(
                    f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条"
                )

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
                    lambda x: 0 if "BENIGN" in x else 1
                )

                # 显示标签分布
                benign_count = sum(data[last_column] == 0)
                attack_count = sum(data[last_column] == 1)
                print(
                    f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条"
                )
            if (
                "dataset\\CIC-IoT-Dataset-2023" in data_filepath
                or "dataset/CIC-IoT-Dataset-2023" in data_filepath
            ):
                # 假设最后一列是标签列
                last_column = data.columns[-1]
                print(
                    f"处理CIC-IoT-Dataset-2023数据集，将最后一列'{last_column}'编码为二分类"
                )

                # 检查类型并转换为字符串
                if data[last_column].dtype != "object":
                    data[last_column] = data[last_column].astype(str)

                # 将包含"BENIGN"的记录编码为0，其他编码为1
                data[last_column] = data[last_column].apply(
                    lambda x: 0 if "BENIGN" in x else 1
                )

                # 显示标签分布
                benign_count = sum(data[last_column] == 0)
                attack_count = sum(data[last_column] == 1)
                print(
                    f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条"
                )
            if (
                "dataset\\UNSW-NB15" in data_filepath
                or "dataset/UNSW-NB15" in data_filepath
            ):
                # 获取标签列
                last_column = data.columns[-1]
                print(
                    "UNSW-NB15数据集，0 for normal and 1 for attack records, 无需处理"
                )

                # 确保标签列为字符串类型
                data[last_column] = data[last_column].astype(str)

                # 使用向量化操作计算标签分布
                benign_count = sum(data[last_column] == "0")
                attack_count = sum(data[last_column] == "1")
                print(
                    f"标签分布: 正常流量(0)={benign_count}条, 攻击流量(1)={attack_count}条"
                )
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
                            label_encoders[column].transform(
                                label_encoders[column].classes_
                            ),
                        )
                    )
                    # print(f"编码前后对照: {encoding_map}")
            tran_data = data.values
            tran_data = np.nan_to_num(tran_data, nan=0.0, posinf=0.0, neginf=0.0)
            scaler = MinMaxScaler()
            tran_data_scalSSC = scaler.fit_transform(tran_data)
            tran_data_scalDFNO = scaler.fit_transform(tran_data)

            for k in tqdm.tqdm(k_list, desc="k values", ncols=120, leave=False):
                for Alpha in tqdm.tqdm(
                    Alpha_list, desc="Alpha values", ncols=120, leave=False
                ):
                    y_true = tran_data[:, -1]
                    subspaceAnomalyScore, affinity_matrix = SSCOMPOD(
                        data=tran_data_scalSSC
                    )
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

                    # 计算AUC和保存ROC曲线
                    auc_results = calculate_auc_and_save(
                        y_true, scores_dict, dataset_name=data_filepath
                    )

                    # 保存到文件
                    # print(f"保存结果到文件: {dataset_name}")
                    with open(
                        f"./results/{dataset_name}.txt", "a", encoding="utf-8"
                    ) as f:
                        f.write(f"\n===== {dataset_name} =====\n")
                        f.write(f"Parameters: k={k}, Alpha={Alpha}\n")
                        for method, auc_value in auc_results.items():
                            f.write(f"{method}: {auc_value:.4f}\n")
                            # 如果是nan，抛出异常
                            if np.isnan(auc_value):
                                raise ValueError(
                                    f"数据集 {dataset_name} 的 {method} 计算结果为 NaN"
                                )
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {e}")
            continue


if __name__ == "__main__":
    main(dataset_dir="dataset/UNSW-NB15")
    main(dataset_dir="dataset/CIC-IDS-2017/MachineLearningCVE")
    # main(dataset_dir="dataset/kddcup.data")
    # main(
    #     dataset_dir="dataset/CIC-IoT-Dataset-2023"
    # )  # This dataset contains only only attack records, so it is not suitable.

    # main(dataset_dir="dataset/maple", file_threshold=150)   # Unfortunately, the dataset is unlabeled.
