# 开发文档（面向接手开发者）

本项目的目标、架构、核心模块、HTTP 接口与约定，帮助新的开发者快速接手并安全地继续迭代。

## 1. 项目概述

**打码工作台**：基于 NudeNet（YOLOv8 ONNX）的本地 AI 打码工具，支持图片 / GIF / 视频。所有推理与编码在**本机**完成，不上传任何数据。

| 项 | 值 |
|---|---|
| 语言 / 环境 | Python 3.9+（开发用 3.10），Windows 为主 |
| 运行依赖 | numpy、opencv-python、onnxruntime、Pillow、imageio-ffmpeg |
| 检测模型 | NudeNet 640m.onnx（约 100MB，启动时自动下载）/ 回退 320n.onnx |
| 前端 | 纯 HTML/JS 单页（嵌在 webui.py 的 `INDEX_HTML` 字符串里），无框架、无构建 |
| 后端 | Python 标准库 `http.server`（ThreadingHTTPServer），无框架 |
| 协议 | AGPL-3.0（因上游 NudeNet 为 AGPL） |

### 文件职责

| 文件 | 职责 | 重要程度 |
|---|---|---|
| `webui.py` | 网页服务 + 前端全部逻辑（单文件前后端） | 核心，最大 |
| `censor_core.py` | 检测核心：模型加载、推理、NMS、打码渲染、长图切片 | 核心 |
| `media_core.py` | GIF / 视频逐帧处理与编码 | 核心 |
| `auto_censor.py` | 命令行入口 | 辅助 |
| `check_deps.py` | 依赖检测与自动安装 | 辅助 |
| `selftest.py` | 自动化自测（73 项断言） | 质量门 |
| `启动打码工作台.bat` | Windows 一键启动 | 辅助 |

---

## 2. 快速上手

```bash
# 安装依赖（有 check_deps 可自动装）
pip install numpy opencv-python onnxruntime Pillow imageio-ffmpeg

# 启动网页版（默认 8080；--lan 允许局域网访问）
python webui.py 8080 --lan

# 命令行打码
python auto_censor.py 图片.jpg --mode mosaic --strength 35
```

**务必运行自测**：`python selftest.py`（73 项断言，覆盖检测/渲染/接口/媒体任务），提交前必须全绿。

---

## 3. 核心模块

### 3.1 censor_core.py —— 检测与打码核心

**常量**
- `MODEL_SOURCES`：640m 模型下载源列表（优先本仓库 Release 附件，其次 hf-mirror 镜像）
- `LABELS`：18 个 NudeNet 类别（见下）
- `DEFAULT_CLASSES`：默认打码类别（仅 5 个 `*_EXPOSED` 隐私部位）——**注意**：检出但不在 `DEFAULT_CLASSES` 里的不会被打码，前端可勾选
- `CENSOR_MODES = ("mosaic", "blur", "solid", "img")`：四种打码方式
- `SLICE_RATIO = 2.5`、`SLICE_MIN_SIDE = 480`：长图切片触发阈值

**NudeNet 18 类**（顺序即模型输出通道索引 4..21）：
`FEMALE_GENITALIA_COVERED, FACE_FEMALE, BUTTOCKS_EXPOSED, FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED, ANUS_EXPOSED, FEET_EXPOSED, BELLY_COVERED, FEET_COVERED, ARMPITS_COVERED, ARMPITS_EXPOSED, FACE_MALE, BELLY_EXPOSED, MALE_GENITALIA_EXPOSED, ANUS_COVERED, FEMALE_BREAST_COVERED, BUTTOCKS_COVERED`

**关键函数**

- `ensure_model()` → 模型路径：存在 640m 则用之；否则按 `MODEL_SOURCES` 逐个下载，全失败回退 320n。下载前经 `_host_allowed` 校验（仅 https、拒绝私网/环回——**改这里注意别引入 SSRF**）
- `class Detector`：
  - `__init__(model_path, inference_resolution=640)`：加载 ONNX，`providers=["CPUExecutionProvider"]`
  - `detect(image, conf=0.25, iou=0.45)` → `[{'class','score','box':[x,y,w,h]}]`。**长图（长宽比>2.5 且短边≥480）自动走 `_detect_sliced`**
  - `_detect_sliced`：切重叠块（20% 重叠，块尺寸 = 短边，限制 480~960）→ 各块检测 → 坐标映射回原图 → `cv2.dnn.NMSBoxes` 去重。**改这个注意坐标映射与越界**
  - 预处理：`cvtColor(RGBA2BGR) + swapRB=True`，与上游 NudeNet 管线一致，**不要"修正"通道顺序**
- `build_cfg(mode, strength, margin, color, classes, conf, asset)`：归一化校验所有打码参数（钳位），返回 cfg dict
- `censor_regions(img, detections, cfg, stamp=None)` → 打码区域数。在 img 上就地修改。`stamp` 是图片遮挡模式的遮挡图（BGR ndarray）

**打码方式实现**（都在 `censor_regions`）：
| mode | 实现 |
|---|---|
| solid | 检测框外扩 margin 后填纯色（BGR） |
| img | 遮挡图等比例放大至完全覆盖框、居中裁剪；无图退化为 solid |
| mosaic | 缩块后最近邻放大（块大小随 `strength` 与图片尺寸缩放） |
| blur | GaussianBlur（核随 strength 与尺寸，偶数自动-1） |

### 3.2 media_core.py —— GIF / 视频

- `_find_ffmpeg()`：优先 `imageio-ffmpeg` 自带二进制，其次系统 PATH。**视频带音频靠它**
- `_default_detect_every(kind, n)`：检测帧距——视频约每秒 6 次、GIF 全程约 10 次；其余帧沿用最近检测框（性能关键，别改成逐帧检测，除非有 GPU）
- `process_gif(data, cfg, detector, stamp, ...)` → `(bytes, "gif", info)`：PIL 逐帧，`info` 含 `frames/detect_every`
- `process_video(data, cfg, detector, stamp, detect_every, progress, prefer)` → `(bytes, ext, info)`：
  - 有 ffmpeg → **mp4 / H.264 + AAC（带音频）**，音轨从输入视频映射（`-map 1:a:0?`）
  - 无 ffmpeg → 按 `prefer` 输出 webm(VP80) 或 mp4(mp4v)，均无音频
  - `progress(done, total)` 回调驱动前端进度条
  - 路径安全：输入统一写临时目录 `in.bin`，**客户端文件名/扩展名不进入任何路径**

### 3.3 webui.py —— 网页服务 + 前端

**HTTP 端点**

| 方法 & 路径 | 功能 |
|---|---|
| GET `/` | 返回单页 HTML（`INDEX_HTML`） |
| POST `/infer?<cfg>` | 图片打码，body=图片字节。返回 `{detections, censored_count, elapsed_ms, fmt, censored(dataURL)}` |
| POST `/asset` | 上传图片遮挡素材，返回 `{id, w, h}`（内存缓存，`ASSET_KEEP=8` 个） |
| POST `/video?kind=gif|video&<cfg>` | 提交媒体任务，body=文件字节 → `{id, kind}` |
| GET `/video/status?id=` | 轮询进度 `{st, pct, frame, frames, info}` |
| GET `/video/result?id=` | 下载结果文件（GIF/MP4/WebM） |
| GET/POST `/lan?mode=on|off` | 查询/切换局域网模式（切换用 `os.execv` 重启自身） |

**cfg 查询参数**：`mode strength margin color conf res classes(逗号分隔) fmt(jpg|png) quality vfmt(webm|mp4) asset(id)`

**前端结构**（`INDEX_HTML` 的 `<script>` 内）：
- 主题：默认跟随系统 `prefers-color-scheme`，手动切换写 localStorage `cb-theme`
- 设置：localStorage `cb-settings`（`SET` 对象：`imgFmt/jpgQuality/vidFmt/suffix/autoDl/remember/ai`）
- 队列：`jobs[]`，每项 `{file, st(pending/run/done/fail), media, kind, dim, origUrl, censUrl, ...}`
- 图片走同步 pump（`pump()`），GIF/视频走后台任务 + `pollMedia` 轮询（600ms）
- 手动模式画布：`edPaint` canvas 叠加在 `edBase` 上；橡皮用 `destination-out` 只擦手绘层
- **灯箱**：`#lightbox`，点卡片打开
- **打包下载**：`buildZip()` 零依赖 STORE 格式 zip 生成器（含 CRC32），点"打包下载全部"生成

**后端注意点**
- `MEDIA` / `ASSETS` 是内存 dict，进程重启即丢（可接受，结果文件在临时目录，`MEDIA_KEEP=6` 自动清理）
- `MEDIA_LOCK` 是 `threading.Lock`（**不可重入**）——`_media_snapshot` 设计为调用方持锁时调用，别在内部再 `with MEDIA_LOCK`
- `CURRENT_PORT` / `CURRENT_LAN` 是模块级变量，`__main__` 里赋值，供 `/lan` 端点读取
- `/lan` 切换用 `os.execv` **替换当前进程**（不返回），响应需先 flush 再 execv

---

## 4. 约定与安全底线

1. **SSRF**：任何对外请求 URL 必须过校验（见 `censor_core._host_allowed`）。本仓库的下载源、发布脚本都是代码内常量 + 域名白名单
2. **路径安全**：客户端文件名 / 扩展名**不得**进入文件系统路径。视频输入固定写 `in.bin`，asset id 是 uuid hex。`.gitignore` 排除了模型与测试图
3. **预处理通道顺序**：`RGBA2BGR + swapRB` 是模型验证管线，勿改
4. **锁**：`MEDIA_LOCK` 不可重入
5. **长图切片**：改 `_detect_sliced` 务必回归 `selftest.py` 的 3b 节（坐标在图界内、普通图不误触发）
6. **前端无构建**：改 `webui.py` 里的 JS 后无需编译，刷新页面即可；但**服务需重启**才加载新的 `INDEX_HTML`

---

## 5. 发布新版本流程

1. 改代码 → `python selftest.py` 全绿
2. `git commit`、`git push origin main`
3. 打 tag：`git tag vX.Y.Z && git push origin vX.Y.Z`
4. 打 zip：`git archive --format=zip --output=ai-auto-censor-vX.Y.Z.zip vX.Y.Z`
5. 建 Release 并上传附件（zip + 640m.onnx + 320n.onnx）：
   - 用本仓库 `_make_release.py` 同款逻辑（读取 GCM 里已存的 GitHub token，仅内存使用）
   - 或 GitHub 网页手动：Create release from tag → 拖入附件
6. 验证：`https://github.com/<owner>/<repo>/releases/latest/download/<附件>` 匿名可达

> 历史发布脚本在发布后已删除，可随时按本文件第 5 节重建（结构见本仓库 README 与 git 历史）。

---

## 6. 已知边界 / 待办

- **长图切片**：检测块沿用"最近检测框"策略，目标横跨切片边界时可能漏（重叠已缓解但不完全）。若要更稳可加跨块 IoU 融合
- **视频性能**：CPU 逐帧打码较慢（每秒约 6 次检测），长视频耗时长；未来可接 GPU provider
- **内存**：`MEDIA` / `ASSETS` 内存态，重启即清；大量任务时临时目录结果会被 `MEDIA_KEEP` 清理
- **局域网**：暴露 0.0.0.0 时无鉴权，仅建议可信网络使用
- **前端测试**：目前靠 `selftest.py` 的接口级断言 + 手动浏览器验证，未接入自动化浏览器测试

## 7. 常见问题速查

| 现象 | 原因 / 处理 |
|---|---|
| 视频没声音 | 确认已装 `imageio-ffmpeg`（`check_deps.py` 会自动装）；卡片预览已不再强制静音，点播放即有声 |
| 长图 0 检测 | 已由切片修复；确认代码是最新版（v1.1.0+） |
| 端口被占用 | `启动打码工作台.bat` 会自动关掉旧 webui 实例；`python webui.py <端口>` 换端口 |
| 局域网访问不了 | 用 `--lan` 或设置开关；确认 Windows 防火墙放行该端口 |
| 模型下载慢/失败 | 自动按顺序尝试"Release 附件→hf-mirror"；仍失败手动下载 640m.onnx 放程序目录 |
