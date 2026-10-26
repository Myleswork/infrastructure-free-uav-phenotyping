# 水稻黄叶识别诊断输出

在项目根目录执行（使用已安装项目依赖的 Python 环境）：

```bash
.venv/bin/python code/yellow_leaf_detection/predict_orthomosaic.py
```

`main()` 保留当前默认输入、模型路径及 `tile_size=2048, overlap=256`。
输入是 `data/2025-08-27/Full Orthomosaic (10 meter).png`，SAM 权重是
`code/yellow_leaf_detection/weights/sam_vit_b.pth`。SegVeg 仍按原顺序通过
`pickle.load` 加载 `SegVeg_data/model_scikit`、`SegVeg_data/XGBoost`、
`SegVeg_data/Necrosis.pkl`，并保留 XGBoost 兼容属性设置。
导入脚本中的函数不会加载权重或启动推理。

## 输出

所有 PNG 与输入影像宽高一致，结果写入 `results/`，再次运行会更新这些结果文件。

| 文件 | 含义 |
| --- | --- |
| `plant_mask.png` | 单通道 0/255；白色是 `generate_plant_mask()` 的植被输出，在分类函数修改它之前复制，独立拼接。保留原有 SAM 选掩膜、形态学清理和缩放处理；并非从最终类别反推。 |
| `yellow_mask.png` | 单通道 0/255；白色是经过原有 HSV 后处理、再拼接后的最终类别 2（senescent vegetation）。 |
| `abnormal_overlay.png` | 完整 RGB 底图；仅在 `yellow_mask` 的白色像素以 0.5 权重混合 RGB 红色 `[255, 0, 0]`。其他像素逐像素保持原色与亮度。 |
| `Disease Detection.png` | 保留原有类别配色和全图混合展示，修正 RGB/BGR 边界。该旧式展示仍会压暗背景；直接对照原图请查看 `abnormal_overlay.png`。 |
| `summary.json` | 模型输出像素计数、百分比、统计定义、拼接规则及解释范围；终端同时打印统计。 |

## 统计口径

- `total_pixels = image_width * image_height`，包含输入影像中所有像素。
- `sam_vegetation_pixels` 是独立拼接 SAM 掩膜的非零像素数；重叠区按并集只统计一次。
- `green_pixels`、`yellow_pixels` 分别来自 HSV 后最终拼接分类掩膜的类别 1、2。
- `other_pixels = total_pixels - green_pixels - yellow_pixels`，包括背景及被 HSV 剔除的像素，因此不能简单解释为非植被面积。
- `vegetation_coverage_percent = 100 * sam_vegetation_pixels / total_pixels`。
- `yellow_ratio_percent = 100 * yellow_pixels / (green_pixels + yellow_pixels)`。分母为最终有效分类植被像素数，并非 SAM 植被像素数；分母为 0 时输出 0.0，终端会说明无有效分类植被。
- 所有计数都在拼接后的整图上计算，避免按切片重复计数。

这些是模型识别结果统计，不能等同于真实病害发生率或真实长势异常率。
没有使用地理分辨率计算平方米或亩数。

## 验证

```bash
.venv/bin/python -m py_compile code/yellow_leaf_detection/predict_orthomosaic.py code/yellow_leaf_detection/test_predict_orthomosaic.py
.venv/bin/python -m unittest discover -s code/yellow_leaf_detection -p 'test_*.py' -v
```

测试使用手工可核对的小尺寸图像，覆盖掩膜输出、边缘切片与重叠冲突、分类后 SAM
掩膜独立性、HSV 剔除、统计分母、零植被、RGB 叠加、文件尺寸及异常输入。
测试不加载实际模型权重，也不运行 SAM/GPU 推理；流程测试只替换 SAM 输出及分类器概率来源，
保留真实特征提取、分类阈值、HSV、拼接和输出逻辑。

检查实际结果时，可以对比原图与 `abnormal_overlay.png`；红色标记应严格对应
`yellow_mask.png` 的白色像素。检查 JSON 时，绿色、黄化、其他三项之和应等于总像素数，
`yellow_mask` 白色像素数应等于 `yellow_pixels`。

## 保留的行为与后续诊断建议

1. `stitch_tiles()` 仍使用 `np.maximum()`。SAM 的二值掩膜合并等价于并集；类别冲突时
   2 覆盖 1，可能使黄化统计偏高。下一步可先统计重叠区分类冲突，并用人工标注验证
   中心区域优先或概率融合等策略；本次未更换拼接算法。
2. `hsv_filter_senescent()` 的内部转换声明为 BGR，但现有调用传入 RGB，本次保留。
   当前仅使用饱和度 S 与亮度 V，交换 R/B 不影响 S/V，故该不一致目前不改变过滤结果；
   若未来使用 H 分量，应先统一通道约定并验证分类变化。
3. 仍使用 SAM ViT-B、现有 SegVeg 权重、分类调用中的 `threshold=0.8` 及原 HSV 阈值。
   模块内遗留的 `XGBOOST_THRESHOLD=0.9998` 未被本流程使用，也未修改。
4. 植被漏检、误检及黄化分类误差需要标注验证；新增诊断图有助于定位这些问题，
   本次没有改进或评估模型精度，没有训练新模型。
5. 旧权重反序列化时可能提示 scikit-learn 版本不一致及 `pkg_resources` 弃用警告。
   本次保留加载方式和依赖版本；警告不等于已验证模型在不同依赖版本间完全等价。
