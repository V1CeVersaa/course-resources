#!/bin/zsh

echo "正在清理 __pycache__ 目录..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "正在清理 .pyc 文件..."
find . -name "*.pyc" -delete

echo "清理完成！"
