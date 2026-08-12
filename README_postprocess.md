# pySTAMPS SBAS Stage 7/8 后处理脚本

适用数据：

- `ps2.mat`
- `mean_v.mat`
- `scla_smooth2.mat`
- `phuw_sm2.mat`
- `uw_space_time.mat`
- `stage7_sbas_debug.json`
- `stage8_sbas_debug.json`

默认数据集：

```text
/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized
```

## 1. 安装依赖

不要重新执行 `conda install` 修改当前 `stamps` 环境中的 NumPy。

先检查：

```bash
/home/ubuntu/software/miniconda3/envs/stamps/bin/python - <<'PY'
import numpy
import scipy
import pandas
import matplotlib
import h5py
import rasterio
import pyproj
print("dependency check passed")
PY
```

`geopandas`和`pyarrow`只用于GeoPackage和Parquet；缺失时脚本会跳过对应格式。

## 2. 放置脚本

```bash
cp pystamps_sbas_postprocess.py /home/ubuntu/software/pystamps-main/
cp run_postprocess_all.sh /home/ubuntu/software/pystamps-main/
cp export_all_epoch_tifs.sh /home/ubuntu/software/pystamps-main/

chmod +x \
  /home/ubuntu/software/pystamps-main/pystamps_sbas_postprocess.py \
  /home/ubuntu/software/pystamps-main/run_postprocess_all.sh \
  /home/ubuntu/software/pystamps-main/export_all_epoch_tifs.sh
```

## 3. 推荐运行

生成点矢量、速度栅格、SCLA栅格、HDF5时序和PNG图：

```bash
cd /home/ubuntu/software/pystamps-main
./run_postprocess_all.sh
```

查看：

```bash
tmux -L postprocess attach -t cangzhou_postprocess
```

## 4. 输出目录

```text
postprocess/
├── points/
│   ├── ps_velocity.csv
│   ├── ps_velocity.gpkg
│   ├── ps_velocity.parquet
│   └── ps_velocity_sample.kml
├── rasters/
│   ├── geo_velocity.tif
│   ├── geo_temporal_rms_mm.tif
│   ├── geo_scla_k_rad_per_m.tif
│   ├── geo_scla_c_rad.tif
│   ├── geo_ps_count.tif
│   └── wgs84/
├── timeseries/
│   └── ps_timeseries.h5
├── plots/
│   ├── 01_velocity_map.png
│   ├── 02_temporal_rms_map.png
│   ├── 03_scla_k_map.png
│   ├── 04_velocity_histogram.png
│   ├── 05_representative_timeseries.png
│   └── 06_valid_ps_by_epoch.png
├── postprocess_report.json
└── postprocess_manifest.csv
```

## 5. 导出全部日期GeoTIFF

257期栅格会占用较多空间。确认磁盘后执行：

```bash
cd /home/ubuntu/software/pystamps-main
./export_all_epoch_tifs.sh
```

仅每5期输出一景：

```bash
/home/ubuntu/software/miniconda3/envs/stamps/bin/python \
  /home/ubuntu/software/pystamps-main/pystamps_sbas_postprocess.py \
  --dataset /mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized \
  --resolution-m 50 \
  --vector-formats csv \
  --write-hdf5 \
  --epoch-tifs \
  --epoch-step 5 \
  --reference-mode existing \
  --overwrite
```

生成结果命名为：

```text
geo_timeseries/geo_YYYYMMDD.tif
```

## 6. 外部参考区域

Stage 8已经进行了内部参考。默认：

```text
--reference-mode existing
```

不再二次校正。

指定经纬度500米参考区：

```bash
python pystamps_sbas_postprocess.py \
  --dataset /path/to/dataset \
  --reference-mode point \
  --ref-lon 116.8 \
  --ref-lat 38.3 \
  --ref-radius-m 500 \
  --write-hdf5 \
  --overwrite
```

指定经纬度矩形：

```bash
--reference-mode bbox \
--ref-bbox 116.70 38.20 116.75 38.25
```

重新参考遵循：

```text
V_new = V_old - median(V_ref)
D_new(t) = D_old(t) - median(D_ref(t))
```

## 7. 正负号

输出采用：

```text
正值：朝向卫星
负值：远离卫星
```

相位转LOS位移：

```text
D_mm = -phase_rad × wavelength_m / (4π) × 1000
```

## 8. 栅格化原则

默认50米投影网格，自动选择当地UTM坐标系。

每个栅格像元取其中PS点的均值，不进行克里金、IDW或最近邻填补，避免制造没有观测支持的连续面。
