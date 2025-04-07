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


if __name__ == "__main__":
    import os
    file_path_list = [os.path.join("results", f) for f in os.listdir("results") if f.endswith(".txt")]
    # for file_path in file_path_list:
    #     analyze_results(file_path)
    for file_path in file_path_list:
        find_best_results(file_path)
