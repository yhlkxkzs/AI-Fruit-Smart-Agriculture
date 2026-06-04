# production_lite0_singlehead_10class

**用途**：App 主模型槽位（`github_model_id`: **`deploy_efficientnet_lite0`**）。

| 项 | 值 |
|----|-----|
| 架构 | EfficientNet-Lite0 **单头** |
| 类别 | **10** 个品种（当前权重，2026-05） |
| GitHub | `best.pt` + `classes.json` **已推送** |

## 2026-06-04

- Multistate **275 类** 对比模型在 `exports/efficientnet_lite0/` 等目录；本槽位仍为旧 10 类，待选定 backbone 后复制替换：

```bash
cp exports/efficientnet_lite0/best.pt exports/production_lite0_singlehead_10class/best.pt
cp exports/efficientnet_lite0/classes.json exports/production_lite0_singlehead_10class/classes.json
cd tasks/fruit_classification && git add exports/production_lite0_singlehead_10class/ && git commit && git push
```

历史 11 类双头：`../_archive/multistate_v1_20260527_lite0_11species/`
