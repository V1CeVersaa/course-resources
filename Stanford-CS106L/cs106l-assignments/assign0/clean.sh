#!/bin/bash

# 清理脚本 - 用于删除编译生成的目标文件和 autograder 中的虚拟环境

echo "开始清理..."

find . -type f -name "*.o" -delete
find . -type f -name "*.out" -delete
find . -type f -name "main" -delete
find . -type f -perm /111 -not -name "*.sh" -not -path "*/\.*" -deletex


if [ -d "autograder" ]; then
    echo "清理 autograder 虚拟环境..."
    # 删除虚拟环境相关文件和目录
    rm -rf autograder/bin
    rm -rf autograder/lib
    rm -rf autograder/include
    rm -f autograder/pyvenv.cfg
    rm -rf autograder/__pycache__
fi

find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

echo "清理完成！"
