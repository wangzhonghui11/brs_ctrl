import h5py

# 打开H5文件
with h5py.File('collected_data-2025-05-08-13-48-08.h5', 'r') as f:
    # 查看文件结构
    print("文件内容:")


    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")


    f.visititems(print_attrs)

    # 查看数据集内容示例
    if 'dataset_name' in f:
        dataset = f['dataset_name']
        print("\n数据集内容:")
        print(dataset[:])  # 显示实际数据