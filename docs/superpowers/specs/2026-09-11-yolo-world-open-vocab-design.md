# YOLO-World 开放词汇检测设计

日期：2026-09-11
状态：已确认（用户批准）

## 目标

在现有 NudeNet 18 类隐私部位检测之外，接入 YOLO-World 开放词汇检测，让用户输入**任意英文类别词**（如 `gun, knife, face`）即可自动检测并打码。未使用自定义类别时，现有功能零影响。

## 方案（用户选定：动态任意类别）

两阶段 prompt-then-detect 架构，纯 onnxruntime + numpy + cv2，无 torch/ultralytics：

1. `tokenizers` 库用 CLIP BPE 把类别词编码成 [K,77] int64
2. CLIP ViT-B/32 文本塔（int8 量化 ONNX，64.5MB）输出 [K,512]，L2 归一化 → txt_feats
3. YOLO-World-S 检测器（48.8MB，动态 txt_feats 输入）输出 [1, 4+K, 8400]
4. numpy 阈值过滤 + xywh→xyxy + `cv2.dnn.NMSBoxes` → 与 NudeNet 检测结果合并 → 现有打码渲染

## 模型来源（双源，hf-mirror 优先，国内可用）

| 文件 | 大小 | 源 1（镜像） | 源 2（主站） |
|---|---|---|---|
| yolov8s-worldv2.onnx | 48.8MB | hf-mirror.com/Instemic/yolo-world-onnx/resolve/main/yolov8s-worldv2.onnx | huggingface.co/Instemic/yolo-world-onnx/... |
| text_model_quantized.onnx | 64.5MB | hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx | huggingface.co/Xenova/clip-vit-base-patch32/... |
| tokenizer.json | 2.2MB | hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json | 同上 |

下载沿用 `censor_core._host_allowed` SSRF 防护；下载到程序目录 `yolo_world/` 子目录。
模型仅在首次使用时下载（懒加载）。

## 模块设计

### yolo_world.py（新文件，~250 行）

- `TextEncoder`：加载 tokenizer.json + 文本塔 ONNX。`encode(words) -> np.ndarray [K,512] float32, L2 归一化`。pad_id=49407，起始 49406，context 77。
- `WorldDetector`：懒加载检测器 session（进程内常驻）。`detect(image, classes, txt_feats, conf, iou) -> [{'class','score','box'}]`，格式与 `censor_core.Detector.detect` 完全一致。class 标识用 `WORLD:<word>` 前缀与 NudeNet 类隔离（如 `WORLD:gun`）。预处理：letterbox 到 640、/255、RGB（cv2 完成）；后处理：`output0[0].T` → [8400, 4+K]，前 4 通道已解码 xywh（输入尺度），后 K 通道已 sigmoid。
- 嵌入缓存 `yolo_world/embeds.npz`：`(类名元组字符串) -> [K,512]`。命中缓存则文本塔完全不加载。类别列表（含顺序）变化即视为新 key。
- `ensure_models()`：检测器与 tokenizer/文本塔缺失时按上表顺序下载，失败抛出带明确提示的异常。
- 长图：复用 `censor_core.Detector._detect_sliced` 的切片思路——`WorldDetector.detect` 内部对长图（同 SLICE_RATIO 阈值）走同样的切块 + 坐标映射 + NMS 流程。

### censor_core.py（改动极小）

`build_cfg` 新增参数 `world_classes`：None 或字符串列表，每项做 trim/去空/lower、去重、限长（每词 ≤64 字符、最多 16 个词），原样保存进 cfg（自定义词不在 LABELS 里，**跳过**现有 `c in LABELS` 过滤）。`censor_regions` 零改动——它只按 cfg["classes"] 匹配 d["class"]。

### webui.py

- cfg 查询参数新增 `world`（URL-encoded 逗号分隔字符串）。
- `do_POST /infer` 与 `/video`：若 `world` 非空 → 懒加载 WorldDetector（模块级单例，线程锁保护），跑 `ensure_models()` + 嵌入编码（缓存命中则毫秒级）+ 检测，结果与 NudeNet 检测合并后再交 `censor_regions`；cfg["classes"] 扩展为 `原勾选 + WORLD:* 命中类别`。为保持现有并发语义，WorldDetector 调用点复用现有单线程队列/pump 逻辑（视频 worker 亦单线程）。
- 前端 `INDEX_HTML`：类别勾选区下方新增输入框（placeholder："自定义类别，英文逗号分隔，如 gun, knife"）+ 一行说明（中文输入无效，需英文）。`readSettings()` 带上 `world`；"记住打码设置"兼容存储。

### auto_censor.py

新增 `--world "gun,knife"` 参数，走同一 cfg 与检测合并路径。

### check_deps.py

依赖清单加 `tokenizers`。

## 参数与行为

- 置信度：复用现有"阈值"滑块（同一 conf 传给两个检测器）。YOLO-World 对训练集外词分数偏低，内部下限 0.04 不变。
- 图片：全图检测，长图自动切片（与 NudeNet 同规则）。
- GIF/视频：与现有节奏一致（约每秒 6 次检测），检测帧上跑双检测器，非检测帧沿用最近框（dets 列表合并后统一沿用，media_core 零改动）。
- 嵌入编码：每次任务开始时对当前词列表取缓存或编码一次（非每帧）。

## 错误处理

- 模型下载失败：HTTP 400 返回明确中文错误"YOLO-World 模型下载失败，请检查网络后重试"，任务标记失败，不影响其他任务。
- `tokenizers` 缺失：提示"pip install tokenizers"。
- 中文/无效词：CLIP 编不出有意义嵌入 → 照常运行，检不出即不打码，不报错。输入框前端 trim 空段。

## 测试

`selftest.py` 新增断言（目标 +10 项）：
1. `WORLD:gun` 类标识生成与 cfg 校验（trim/去重/限长）
2. 嵌入缓存：同词列表二次编码命中缓存（文本塔不加载）
3. WorldDetector 输出格式与 censor_regions 兼容（用随机权重替代真实模型做形状级测试）
4. 合并检测后 censor_regions 打码计数正确
5. 模型下载源 URL 通过 `_host_allowed`
6. cfg 未启用 world 时行为与现状完全一致（回归）

交付前真机浏览器端到端验证：上传图片 + 输入 gun/face 类别，确认检测与打码生效。**不上传 GitHub，用户先测。**

## 已知边界

- CLIP 是英文塔，中文词无效（UI 有提示）。
- 首次使用需下载约 115MB 模型。
- CPU 每帧检测增加约 100~300ms；未使用自定义类别时零开销。
- `WORLD:*` 类别不进入"记住打码设置"的勾选恢复（输入框内容会记住）。
