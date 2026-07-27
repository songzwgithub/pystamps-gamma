#!/bin/bash
# ============================================================
# push.sh —— 改完代码后一键提交并推送到 GitHub
# 用法：  ./push.sh "你的提交说明"
#        如果不传参数，会用默认说明 "Update code"
# ============================================================

set -e  # 任何一步失败就立即停止

# ---------- 1. 进入脚本所在目录（确保路径正确）----------
cd "$(dirname "$0")"

# ---------- 2. 检查是不是 git 仓库 ----------
if [ ! -d .git ]; then
    echo "❌ 当前目录不是 Git 仓库，请先 git init"
    exit 1
fi

# ---------- 3. 获取提交说明 ----------
if [ -z "$1" ]; then
    COMMIT_MSG="Update code"
    echo "ℹ️  未提供提交说明，使用默认: $COMMIT_MSG"
else
    COMMIT_MSG="$1"
fi

# ---------- 4. 显示当前状态 ----------
echo ""
echo "📂 当前分支: $(git branch --show-current)"
echo "📝 提交说明: $COMMIT_MSG"
echo ""
echo "--- git status ---"
git status --short
echo "------------------"
echo ""

# ---------- 5. 确认是否继续 ----------
read -p "确认提交并推送？(y/n) " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ 已取消"
    exit 0
fi

# ---------- 6. 暂存 → 提交 → 推送 ----------
echo ""
echo "📦 暂存所有改动..."
git add .

echo "💾 提交中..."
git commit -m "$COMMIT_MSG"

echo "🚀 推送到远程..."
git push origin "$(git branch --show-current)"

echo ""
echo "✅ 完成！代码已推送到 GitHub 🎉"