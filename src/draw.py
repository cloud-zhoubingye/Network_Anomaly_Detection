import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

def calculate_auc_and_save(y_true, scores_dict, dataset_name, output_dir="results", filename=None):
    """
    计算各方法的AUC值，并保存到文件
    
    参数:
    y_true: 真实标签
    scores_dict: 字典，键为方法名，值为异常分数
    dataset_name: 数据集名称
    output_dir: 输出目录
    filename: 输出文件名(如果为None，则自动根据数据集名称生成)
    """
    # 提取数据集名称(如果dataset_name包含路径)
    dataset_name = os.path.splitext(os.path.basename(dataset_name))[0]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理文件名
    if filename is not None:
        # 提取filename中的纯文件名部分
        filename = os.path.basename(filename)
    else:
        # 根据数据集名称生成默认文件名
        filename = f"auc_comparison_{dataset_name}.txt"

    # 构建完整输出路径
    output_path = os.path.join(output_dir, filename)

    # 设置ROC图和提升图的文件名
    roc_filename = f"roc_curves_{dataset_name}.png"
    improvement_filename = f"auc_improvements_{dataset_name}.png"

    # 构建图表的完整路径
    roc_image_path = os.path.join(output_dir, roc_filename)
    improvement_image_path = os.path.join(output_dir, improvement_filename)

    results = {}

    # 计算每个方法的AUC
    for method_name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        results[method_name] = roc_auc

    # 找出SCNFOD的结果
    scnfod_key = None
    for key in results.keys():
        if key.startswith("SCNFOD"):
            scnfod_key = key
            break

    # 计算SCNFOD相比其他方法的提升
    improvements = {}
    if scnfod_key:
        scnfod_auc = results[scnfod_key]
        for method, auc_value in results.items():
            if method != scnfod_key:
                abs_improvement = scnfod_auc - auc_value
                pct_improvement = (abs_improvement / auc_value) * 100 if auc_value > 0 else float('inf')
                improvements[method] = (abs_improvement, pct_improvement)

    return results
