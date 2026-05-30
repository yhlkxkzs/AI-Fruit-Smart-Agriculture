#!/bin/bash
# 把所有分类数据源合并成一个 dataset_cls，之后训练只用这一份。
# 以后新加数据：把新数据做成同样结构（train/val/test + 类别子目录），加入 SOURCES 再跑本脚本。

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

# 当前要合并的数据源（可增删）
SOURCES="${SOURCES:-datasetnew_cls datasetlocal_cls}"
# 合并后的唯一输出目录
OUTPUT="${OUTPUT:-dataset_cls}"
# 首次合并可加 --clear；后续追加不要加 --clear
CLEAR="${CLEAR:-}"

echo "合并源: $SOURCES -> $OUTPUT"
ARGS=(--sources $SOURCES --output "$OUTPUT")
[[ -n "$CLEAR" ]] && ARGS+=(--clear)
python train_detection/scripts/merge_cls_datasets.py "${ARGS[@]}"
echo "完成。训练时用: --data $OUTPUT"
