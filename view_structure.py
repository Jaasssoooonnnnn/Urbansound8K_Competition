import os

def print_tree(startpath):
    # 【配置区域】
    # 1. 设置不想深入展开的文件夹（只看第一层）
    shallow_dirs = {'Kaggle_Data', 'urbansound8k'}
    
    # 2. 忽略的文件夹（完全不看，比如隐藏文件夹）
    ignore_dirs = {'.git', '.ipynb_checkpoints', '__pycache__'}
    
    # 3. 防止单一文件夹内文件过多刷屏（如果超过这个数，就折叠显示）
    max_files_per_dir = 15

    print(f"📂 {os.path.basename(os.path.abspath(startpath))}/  (Root)")

    for root, dirs, files in os.walk(startpath):
        # 0. 过滤掉不想看的系统文件夹
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # 计算当前层级
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        subindent = '│   ' * (level + 1)
        
        dirname = os.path.basename(root)
        
        # 打印当前文件夹名称（根目录除外）
        if root != startpath:
            print(f"{indent}├── 📁 {dirname}/")

        # --- 核心逻辑：判断是否是“数据文件夹” ---
        if dirname in shallow_dirs:
            # 如果是数据文件夹：打印里面的东西，然后清空 dirs 以停止递归
            # 打印子文件夹（只打印名字，不进去了）
            for d in dirs:
                print(f"{subindent}├── 📁 {d}/ (不再展开)")
            
            # 打印文件
            file_count = len(files)
            for i, f in enumerate(files):
                if f.startswith('.'): continue
                if i < max_files_per_dir:
                    print(f"{subindent}├── 📄 {f}")
                else:
                    print(f"{subindent}└── ... (还有 {file_count - max_files_per_dir} 个文件被隐藏)")
                    break
            
            # 【关键一步】清空子目录列表，阻止 os.walk 继续向下
            dirs[:] = []
            
        else:
            # --- 普通文件夹逻辑：正常递归 ---
            file_count = len(files)
            displayed_files = 0
            for i, f in enumerate(sorted(files)): # 排序一下更好看
                if f.startswith('.'): continue
                if displayed_files < max_files_per_dir:
                    print(f"{subindent}├── {f}")
                    displayed_files += 1
                else:
                    print(f"{subindent}└── ... (还有 {file_count - max_files_per_dir} 个文件被隐藏)")
                    break

if __name__ == "__main__":
    print_tree(".")
