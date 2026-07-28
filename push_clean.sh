#!/bin/bash
set -e
cd /mnt/h/pystamps-main

if [ ! -d ".git" ]; then
    echo "❌ 不是 Git 仓库，请先 git init"
    exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "🔗 绑定远程仓库..."
    git remote add origin git@github.com:songzwgithub/pystamps-gamma.git
fi

echo "=== [1/5] 扫描 >5M 文件并追加到 .gitignore ==="
find . -type f -size +5M -not -path './.git/*' | sed 's|^\./||' >> .gitignore
sort -u .gitignore -o .gitignore

echo "=== [2/5] 清理 Git 缓存 ==="
git rm --cached '*.stage*_last_backup' 2>/dev/null || true
git rm -r --cached '*.egg-info' 2>/dev/null || true
git rm --cached yes yes.pub id_rsa id_ed25519 2>/dev/null || true

BIG=$(find . -type f -size +5M -not -path './.git/*' | sed 's|^\./||')
if [ -n "$BIG" ]; then
    echo "⚠️  大文件移除跟踪："
    echo "$BIG"
    echo "$BIG" | xargs -I{} git rm --cached "{}" 2>/dev/null || true
fi

echo "=== [3/5] 变更预览 ==="
git status -s

MSG="${1:-Cleanup and update}"
echo ""
echo "📝 提交说明: $MSG"
read -rp "✏️  确认提交并推送？(y/n): " c
[[ "$c" != "y" ]] && echo "已取消" && exit 0

echo "=== [4/5] 提交 ==="
git add .
git commit -m "$MSG"

echo "=== [5/5] 推送 ==="
BRANCH=$(git branch --show-current)
git push -u origin "$BRANCH"

echo ""
echo "✅ 完成！https://github.com/songzwgithub/pystamps-gamma"
