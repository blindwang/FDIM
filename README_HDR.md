# AVS HDR 数据集测试方法

本文只说明 FDIM 在 AVS HDR 数据集上的测试流程，不覆盖其它数据集。

## 1. 环境准备

从仓库根目录安装本地版本：

```bash
pip install -e .
```

确认基础工具可用：

```bash
ffmpeg -version
fdim --help
chmod +x fdim/vmaf/vmaf
```

FDIM 默认 checkpoint 位于：

```text
fdim/dist/checkpoints/dist_5.0.0.ckpt
```

如需覆盖默认模型，可在命令中传入 `--model_path <ckpt_path>`。

## 2. 数据准备

AVS HDR 视频目录按如下结构准备：

```text
<AVSHDR_ROOT>/
  ref_mp4/   # 参考视频
  val_mp4/   # 失真视频
```

仓库内已提供 AVS HDR 的 MP4 元信息 CSV：

```text
data/dataset/AVSHDR-info-mp4-HW.csv
data/dataset/AVSHDR-info-mp4-MLS.csv
data/dataset/AVSHDR-info-mp4-WHU.csv
data/dataset/AVSHDR-info-mp4-ZJU.csv
```

CSV 需要包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `ref_name` | 参考视频文件名 |
| `dis_name` | 失真视频文件名 |
| `mos` | 主观质量分 |
| `trc` | HDR 传输曲线，取值为 `PQ` 或 `HLG` |
| `ref_width`, `ref_height` | 参考视频分辨率 |
| `dis_width`, `dis_height` | 失真视频分辨率 |
| `ref_bits`, `dis_bits` | 视频 bit depth，AVS HDR 通常为 10 |
| `codec` | 编码器名称 |

## 3. HDR 测试口径

AVS HDR 使用 10-bit MP4。测试时：

- `ref_fmt` 和 `dis_fmt` 使用 `rgb`，由 ffmpeg 从 MP4 解码。
- `ref_bit_depth` 和 `dis_bit_depth` 使用 CSV 中的 `ref_bits` / `dis_bits`。
- HDR 深度分支启用 `--preprocess pu21`。
- `display_model` 按 `trc` 或实验室显示条件选择。

标准 HDR TV 口径：

| `trc` | `display_model` |
| --- | --- |
| `PQ` | `standard_hdr_pq_tv` |
| `HLG` | `standard_hdr_hlg_tv` |

实验室显示模型口径：

| 子集 | `trc` | `display_model` |
| --- | --- | --- |
| `HW` | `PQ` | `avs_hdr_set2_pq` |
| `MLS` | `PQ` | `avs_hdr_set3_pq` |
| `WHU` | `PQ` | `avs_hdr_set2_pq` |
| `WHU` | `HLG` | `avs_hdr_set2_hlg` |
| `ZJU` | `PQ` | `standard_hdr_pq_tv` |
| `ZJU` | `HLG` | `standard_hdr_hlg_tv` |

## 4. 单条视频测试

适合先验证环境和单条样本是否能跑通。下面以 HLG 样本为例：

```bash
fdim single \
  --ref <AVSHDR_ROOT>/ref_mp4/Buildings2.mp4 \
  --dis <AVSHDR_ROOT>/val_mp4/Buildings2_AVS3_QP40.mp4 \
  --ref_fmt rgb \
  --dis_fmt rgb \
  --ref_width 3840 \
  --ref_height 2160 \
  --dis_width 3840 \
  --dis_height 2160 \
  --ref_bit_depth 10 \
  --dis_bit_depth 10 \
  --preprocess pu21 \
  --display_model standard_hdr_hlg_tv \
  --input_resolution ori \
  --calc_interval 1 \
  --video_temp_path /tmp/fdim_avshdr_single \
  --gpu_idx 0
```

PQ 样本只需要把 `--display_model` 改成对应 PQ 模型，例如 `standard_hdr_pq_tv` 或实验室显示模型中的 PQ 配置。

## 5. 整集跑批

### 5.1 实验室显示模型口径

该脚本读取 `HW`、`MLS`、`WHU`、`ZJU` 四个子集 CSV，并按子集和 `trc` 自动选择显示模型。
同一 `ref_name` / `dis_name` / `display_model` 只计算一次；不同实验室仅 MOS 不同的行会复用同一组 FDIM/VMAF/deep 结果。
输出仍按实验室逐行保留，便于后续按各实验室 MOS 分别计算相关性。

```bash
.venv-fdim/bin/python -u scripts/run_avshdr_labs_fdim.py \
  --data-root <AVSHDR_ROOT> \
  --labs all \
  --out-dir outputs/avshdr_labs_fdim \
  --temp-root /tmp/fdim_avshdr_labs \
  --gpu-idx 0 \
  --calc-interval 1 \
  --input-resolution ori \
  --n-threads 32
```

输出文件：

```text
outputs/avshdr_labs_fdim/avshdr_labs_fdim.csv
```

## 6. 输出字段

常用输出字段如下：

| 字段 | 含义 |
| --- | --- |
| `case_key` | 输出行唯一键，格式为 `lab:dis_name:display_model` |
| `compute_key` | 实际计算唯一键，格式为 `ref_name:dis_name:display_model`；相同 key 会复用计算结果 |
| `lab` | 实验室子集，取值为 `HW`、`MLS`、`WHU`、`ZJU` |
| `mos` | 当前实验室子集对应的主观质量分 |
| `fdim_score` / `fdim_calc` | FDIM 总分，越高表示预测质量越好 |
| `vmaf_score` / `vmaf_calc` | VMAF 分支原始分 |
| `deep_score` / `deep_calc` | 深度分支分数 |
| `trc` | 当前样本 HDR 传输曲线 |
| `display_model` | 当前样本使用的显示模型 |

## 7. 运行注意事项

- 4K 10-bit HDR 中间 YUV 文件很大，`--temp-root` / `--video_temp_path` 建议放在 `/tmp` 或高速临时盘。
- 跑批脚本会按 `case_key` 逐条写入 CSV；结果文件已存在时会跳过已完成输出行，可直接断点续跑。
- 同一 `compute_key` 已经算过时，脚本会复用缓存分数，只补写其它实验室的输出行。
- 默认不保留临时 YUV；只有排查问题时才使用 `--keep-temp`。
- 如果显存或磁盘压力较大，可把 `--input-resolution ori` 改为 `--input-resolution 1080` 进行低复杂度测试。
