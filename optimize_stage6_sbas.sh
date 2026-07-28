#!/bin/bash

set -e

ROOT=$(pwd)

echo "===================================="
echo "优化 pySTAMPS Stage6 SBAS"
echo "目录:"
echo $ROOT
echo "===================================="


FILE=$(find pystamps -name "stage6_sbas.py" | head -1)

if [ -z "$FILE" ]; then
    echo "错误: 找不到 stage6_sbas.py"
    exit 1
fi

echo "修改文件:"
echo $FILE


cp $FILE ${FILE}.bak_stage6_opt


python - <<PY

from pathlib import Path

p=Path("$FILE")

txt=p.read_text()


# ------------------------------------------------
# 1. 禁止 multiprocessing复制phase
# ------------------------------------------------

old="""
with ProcessPoolExecutor(
        max_workers=io_workers
) as executor:
"""

new="""
# Stage6 optimization:
# 大矩阵禁止多进程复制
# 使用单主进程 + numpy内部线程

with ProcessPoolExecutor(
        max_workers=1
) as executor:
"""

if old in txt:
    txt=txt.replace(old,new)
    print("修改 multiprocessing")


# ------------------------------------------------
# 2. 增大edge chunk
# ------------------------------------------------

txt=txt.replace(
"edge_chunk = 512",
"edge_chunk = 4096"
)


# ------------------------------------------------
# 3. 增加环境线程控制
# ------------------------------------------------

insert="""

# ===== Stage6 performance tuning =====
import os

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "24"
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "24"
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "24"
)

"""

if "Stage6 performance tuning" not in txt:

    txt=insert+txt


# ------------------------------------------------
# 4. numpy线程优化
# ------------------------------------------------

if "threadpool_limits" not in txt:

    txt=txt.replace(
        "import numpy as np",
        """
import numpy as np

try:
    from threadpoolctl import threadpool_limits
except:
    threadpool_limits=None
"""
    )


p.write_text(txt)

print("Stage6修改完成")

PY



echo
echo "===================================="
echo "生成优化启动命令"
echo "===================================="


cat > run_stage6_fast.sh <<'RUN'

#!/bin/bash


export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24


DATASET=/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized


python -m pystamps.pipeline.stage6_sbas \
 --dataset $DATASET \
 --io-workers 1 \
 --edge-chunk 4096 \
 --snaphu-workers 8 \
 --anneal-workers 16 \
 --anneal-runs 5


RUN


chmod +x run_stage6_fast.sh


echo
echo "完成"
echo "备份:"
echo ${FILE}.bak_stage6_opt
echo
echo "运行:"
echo "./run_stage6_fast.sh"

