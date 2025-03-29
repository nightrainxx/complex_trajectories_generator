"""
项目信息收集脚本
收集项目根目录下的所有代码文件，并生成项目结构图
"""
import os
from pathlib import Path
import datetime

def get_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"无法读取文件内容: {str(e)}"

def collect_project_info(root_dir='.', output_file='project_structure.txt'):
    """收集项目信息并生成结构图"""
    # 要忽略的目录和文件
    ignore_dirs = {'.git', '__pycache__', '.pytest_cache', '.vscode', '.idea'}
    ignore_files = {'.gitignore', '.DS_Store'}
    
    # 获取所有Python文件和Markdown文件
    code_files = []
    for root, dirs, files in os.walk(root_dir):
        # 忽略特定目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if file in ignore_files:
                continue
                
            if file.endswith(('.py', '.md', '.txt')):
                code_files.append(os.path.join(root, file))
    
    # 生成输出内容
    output = []
    output.append("# 项目结构报告")
    output.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 生成目录树
    output.append("## 项目结构树")
    output.append("```")
    current_level = 0
    last_dirs = []
    
    for file_path in sorted(code_files):
        rel_path = os.path.relpath(file_path, root_dir)
        parts = rel_path.split(os.sep)
        
        # 处理目录结构
        for i, part in enumerate(parts[:-1]):
            if i >= len(last_dirs) or part != last_dirs[i]:
                output.append("    " * i + "├── " + part)
                last_dirs = parts[:-1]
        
        # 添加文件
        output.append("    " * (len(parts) - 1) + "├── " + parts[-1])
    
    output.append("```\n")
    
    # 添加文件内容
    output.append("## 文件内容")
    for file_path in sorted(code_files):
        rel_path = os.path.relpath(file_path, root_dir)
        output.append(f"\n### {rel_path}")
        output.append("```" + ("python" if file_path.endswith('.py') else ""))
        output.append(get_file_content(file_path))
        output.append("```")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"项目结构和代码已保存到 {output_file}")

if __name__ == "__main__":
    collect_project_info() 