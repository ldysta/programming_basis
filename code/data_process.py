import glob
import pandas as pd
# 加载数据并处理
def data_process(path):
    files = glob.glob(path)
    df_list = []
    # 创建字典储存列信息
    columns_dict = {}
    # 创建字典储存时间戳范围
    timestamp_ranges = {}
    for filename in files:
        df = pd.read_csv(filename)
        # 删除不必要的列
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'index' in df.columns:
            df = df.drop(columns=['index'])  # 删除名为 'index' 的列
        # 确保时间戳列为 datetime 类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # 获取时间戳的最小值和最大值，并规范化时间
        start_time = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
        end_time = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        # 存入字典
        timestamp_ranges[filename] = (start_time, end_time)

        # 删除包含空值的行
        df = df.dropna()

        df_list.append(df)
        # 获取列名并存入字典
        columns_dict[filename] = df.columns.tolist()

    # 打印字典
    for file, cols in columns_dict.items():
        print(f"文件: {file}, 列名: {cols}")
    for file, time_range in timestamp_ranges.items():
        print(f"文件: {file}, 时间范围: {time_range}")

    data = pd.concat(df_list, ignore_index=True)
    print("原始数据标签:",set(data['label']))
    data['label'] = data['label'].astype('category')  # 转换为类别型
    data['label'] = data['label'].cat.codes  # 编码标签
    print("新的数据标签:",set(data['label']))
    print(data.info)

    return data