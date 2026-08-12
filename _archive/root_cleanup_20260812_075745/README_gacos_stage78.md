# pySTAMPS SBAS：GACOS改正后运行Stage 7/8

本补丁在已经完成的Stage 6结果`phuw2.mat`之后加入GACOS改正，再运行自定义SBAS Stage 7和Stage 8。不会重新运行SNAPHU，也不会覆盖原始`phuw2.mat`。

## 支持的GACOS产品

### GeoTIFF

文件名中需包含获取日期，例如：

```text
20210101.ztd.tif
20210101.tif
GACOS_20210101.tiff
```

GeoTIFF必须有正确CRS。默认数值解释为天顶延迟，单位自动读取标签；没有单位标签时按米处理。

### 原始ZTD

```text
20210101.ztd
20210101.ztd.rsc
```

也支持同名`20210101.rsc`。RSC至少需要：

```text
WIDTH
FILE_LENGTH
X_FIRST
Y_FIRST
X_STEP
Y_STEP
```

ZTD按GACOS标准的4字节浮点读取，默认小端、单位米。

## 一键安装和运行

解压后进入脚本包目录：

```bash
chmod +x install_gacos_stage78_pipeline.sh install_stage78_sbas_patch.sh

PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
./install_gacos_stage78_pipeline.sh install-run
```

默认`PYSTAMPS_GACOS_FORMAT=auto`：同一日期若同时存在GeoTIFF和ZTD，优先使用GeoTIFF；其他日期可以使用ZTD，因此允许两类产品混合存在。

只使用GeoTIFF：

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
PYSTAMPS_GACOS_FORMAT=tif \
./install_gacos_stage78_pipeline.sh install-run
```

只使用ZTD：

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
PYSTAMPS_GACOS_FORMAT=ztd \
./install_gacos_stage78_pipeline.sh install-run
```

## 入射角

程序按以下顺序寻找入射角：

1. `PYSTAMPS_GACOS_INCIDENCE_TIF`指定的逐像元入射角GeoTIFF；
2. `PYSTAMPS_GACOS_INCIDENCE_DEG`指定的固定入射角；
3. `ps2.mat/mean_incidence`；
4. `parms.mat/mean_incidence`。

指定固定入射角示例：

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
PYSTAMPS_GACOS_INCIDENCE_DEG=39.5 \
./install_gacos_stage78_pipeline.sh install-run
```

逐像元入射角：

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
PYSTAMPS_GACOS_INCIDENCE_TIF=/路径/incidence_angle.tif \
./install_gacos_stage78_pipeline.sh install-run
```

若输入GeoTIFF已经是LOS延迟而不是ZTD：

```bash
PYSTAMPS_GACOS_PROJECTION=los
```

## 改正符号

默认：

```text
PYSTAMPS_GACOS_SIGN=auto
```

程序从代表性PS和干涉图中比较：

```text
ph_raw - ph_gacos
ph_raw + ph_gacos
```

并选择空间稳健尺度较小的方案。结果记录在：

```text
gacos_correction_debug.json
```

强制减法或加法：

```bash
PYSTAMPS_GACOS_SIGN=subtract
PYSTAMPS_GACOS_SIGN=add
```

## 处理流程

```text
phuw2.mat（原始Stage 6结果，保留）
    ↓ 读取257期GACOS
天顶延迟 / cos(incidence)
    ↓ 使用pySTAMPS相同参考PS
获取日期LOS延迟
    ↓ ifgday_ix：后日期减前日期
763个GACOS干涉相位
    ↓ 自动判定正负号
phuw2_gacos.mat
    ↓
Stage 7 SBAS
    ↓
Stage 8 SBAS
```

## 输出

GACOS专用输出：

```text
phuw2_gacos.mat
gacos_correction_debug.json
gacos_date_inventory.csv
_gacos_work/gacos_los_ref.f32
```

Stage 7/8正常输出：

```text
scla_sb2.mat
phuw_sm2.mat
bp_sm2.mat
scla2.mat
scla_smooth2.mat
mean_v.mat
uw_space_time.mat
stage7_sbas_debug.json
stage8_sbas_debug.json
```

## 监控

```bash
tmux -L stage78gacos attach -t cangzhou_stage78_gacos
```

日志：

```bash
DATASET=/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized
LATEST_LOG=$(
  find "$DATASET/_run_logs" -maxdepth 1 -type f \
    -name 'stage78_gacos_*.log' -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)
tail -f "$LATEST_LOG"
```

## 重新生成改正结果

GACOS产品、入射角或符号设置改变后：

```bash
PYSTAMPS_GACOS_REBUILD=1 \
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
./install_gacos_stage78_pipeline.sh run
```

## 只安装，不运行

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
./install_gacos_stage78_pipeline.sh install
```

前台运行：

```bash
PYSTAMPS_GACOS_DIR=/实际/GACOS/目录 \
./install_gacos_stage78_pipeline.sh foreground
```

## 格式测试

安装后执行：

```bash
cd /home/ubuntu/software/pystamps-main
/home/ubuntu/software/miniconda3/envs/stamps/bin/python \
  /脚本包目录/test_gacos_formats.py
```
