# 全时段联合分段趋势 + 共同季节项 + SBAS协方差 + Huber稳健GLS

## 模型

对每个PS使用全部获取日期联合求解：

```text
phase(t)
= 截距
+ 每个自然年的连续分段线性斜率
+ 所有年份共享的一年周期正弦/余弦项
+ 可选半年周期项
+ 残差
```

这里的“共同季节项”表示：

```text
对同一个PS，季节振幅和相位在所有年份中共享
```

不是让整个研究区所有PS使用同一个季节振幅。

年度趋势基函数在年界处连续，不允许每年独立截距产生人为跳变。

## 默认正式配置

```text
SBAS日期协方差：network
稳健估计：Huber IRLS
季节项：一年周期（sin + cos）
每年至少10景
每年跨度至少240天
模型RMS最高15 mm
年度速率标准误差最高5 mm/yr
```

半年周期不是默认项。需要时使用：

```bash
--seasonal-harmonics 2
```

建议先使用一年周期结果，再把半年周期作为敏感性试验，避免过拟合。

## 一键安装并测试

```bash
cd /home/ubuntu/software/pystamps-main

unzip /上传目录/pystamps_joint_piecewise_seasonal_gls_bundle.zip \
  -d joint_piecewise_seasonal_scripts

cd joint_piecewise_seasonal_scripts
chmod +x install_joint_piecewise_seasonal_gls.sh
./install_joint_piecewise_seasonal_gls.sh install
```

安装器会执行合成数据测试。应显示：

```text
Synthetic joint piecewise-seasonal GLS test passed
```

## 一键运行

```bash
cd /home/ubuntu/software/pystamps-main
./run_joint_piecewise_seasonal_gls.sh
```

查看：

```bash
tmux -L jointtrend attach -t cangzhou_joint_piecewise_seasonal
```

## 输出

```text
joint_piecewise_seasonal_velocity/
├── joint_piecewise_seasonal_velocity.h5
├── joint_piecewise_seasonal_velocity.gpkg
├── joint_year_summary.csv
├── joint_piecewise_seasonal_report.json
├── yearly/                         # 仅使用--write-year-csv时生成
└── _work/                          # 分块断点
```

### HDF5主要变量

```text
velocity_mm_yr                  最终联合稳健GLS年度速率
velocity_gls_mm_yr              未加Huber权重的联合GLS年度速率
velocity_std_mm_yr              年度速率标准误差
ci95_low_mm_yr
ci95_high_mm_yr
recommended                     推荐主结果
strict                          推荐结果+局部一致性
significant                     95%置信区间不跨0
n_obs_year
span_days_year
model_rmse_mm
annual_amplitude_mm
annual_peak_day
```

### QGIS字段

```text
v2021     2021年度速率
se2021    2021年度标准误差
q2021     推荐点
s2021     严格点
sg2021    95%显著点
n2021     年内有效日期数
sp2021    年内时间跨度
ann_amp   共同一年周期振幅，mm
ann_peak  周期峰值日（相对年初）
```

主成果筛选：

```sql
"q2021" = 1
```

显著形变筛选：

```sql
"q2021" = 1 AND "sg2021" = 1
```

## 断点续算

每个PS分块完成后写入：

```text
_work/chunk_*.npz
```

中断后重新运行即可。输入文件或模型参数变化时，旧断点签名自动失效。

## 模型解释

输出年度速率是：

```text
在共同季节项被剥离后，每个自然年内部的连续线性分量
```

它比逐年独立拟合更适合年际比较，因为：

1. 所有年份共同估计季节项；
2. 年界处保持连续；
3. 使用完整SBAS日期协方差；
4. 异常残差模态由Huber权重降低影响。

## 限制

- 每年内部仍假设线性趋势；突发阶跃和年内变速未显式建模。
- 网络协方差默认干涉图误差相互独立。
- 当前实现是针对自定义pySTAMPS SBAS Stage 7/8输出的扩展，不是官方StaMPS逐元素一致性版本。
