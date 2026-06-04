# incoming — App 上传入口

GitHub Actions workflow `afsa-incoming-dispatch.yml` 监听本目录下新图片（`incoming/**/*.jpg` 等），结合同目录 `.afsa.json` sidecar 路由到对应模型。

```text
incoming/
└── category/
    └── uploads/    # 放置待推理图片（勿提交大量用户真实数据）
```

**2026-06-04**：随 multistate 流水线首次纳入仓库；目录结构为空占位，供 App 按契约上传。
