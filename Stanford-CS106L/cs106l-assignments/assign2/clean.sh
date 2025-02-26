#!/bin/zsh

echo "开始清理..."

find . -type f -name "*.o" -delete
find . -type f -name "*.out" -delete
find . -type f -name "main" -delete

if [ -d "autograder" ]; then
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
