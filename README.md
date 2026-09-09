# 打码工作台（AI 自动打码工具）

基于 NudeNet（YOLOv8 ONNX）的本地图片 / GIF / 视频自动打码工具。所有检测与打码**全部在本机完成，图片不会离开你的电脑**。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-AGPL--3.0-green)

## 功能

- **AI 模式**：批量拖入图片 / GIF / 视频，点击"开始打码"自动检测并打码
  - 打码方式：马赛克 / 高斯模糊 / 纯色填充（自定义颜色）/ **图片遮挡**（上传一张图，等比例放大盖住检测区域）
  - 强度、边缘外扩、检测阈值、推理分辨率全部可调
  - 打码类别 18 类可勾选（默认仅隐私部位，脸 / 四肢等按需添加）
  - GIF 逐帧打码；视频逐帧打码（每秒约 6 次检测 + 框跟踪，兼顾速度与召回），实时进度显示
- **手动模式**：画布编辑器，五种画笔（色彩 / 马赛克 / 模糊 / 图片 / 橡皮）手动打码或修补 AI 漏检
  - 可先"AI 预打码"再手动补差；橡皮只擦手动笔迹，不伤 AI 层
  - 支持撤销、多图切换、合成导出
- **通用**：白天 / 夜晚主题（默认跟随系统）、齿轮设置（保存格式 / 质量、视频格式、文件名后缀、自动下载、记住打码设置）、拖拽批量上传、一键下载
- **命令行**：`auto_censor.py` 支持全部打码参数，方便脚本化

## 快速开始

### Windows（推荐）

双击 `启动打码工作台.bat`：

1. 自动检查并安装缺失依赖（首次运行较慢）
2. 缺失模型自动下载（640m 约 100MB，来自国内镜像）
3. 启动服务并自动打开浏览器（默认 `http://localhost:8080`，可传参换端口）

> 局域网访问：`启动打码工作台.bat 8080 --lan` 即可让手机 / 同一局域网内其他电脑访问；
> 也可在网页右上角「设置」→「允许局域网访问」开关直接切换（自动重启生效）。
> 访问地址形如 `http://192.168.x.x:8080`（局域网 IP 会显示在启动日志与设置面板中）。

### 手动安装

```bash
# 1. 安装依赖（Python 3.9+）
pip install numpy opencv-python onnxruntime Pillow
# 国内网络可加清华镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 下载模型（任选其一）
#    主模型 640m（推荐）：
#      https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
#      国内镜像：https://hf-mirror.com/zhangsongbo365/nudenet_onnx/resolve/main/640m.onnx
#    放到项目根目录 640m.onnx。
#    若缺失，启动时也会自动从镜像下载；亦可用回退模型 320n.onnx（同地址改文件名）。

# 3. 启动
python check_deps.py   # 依赖检测与自动安装
python webui.py        # 或 python webui.py 9000 指定端口
python webui.py 8080 --lan   # 允许局域网访问（设置里也能切换）
```

### 命令行打码

```bash
python auto_censor.py 图片.jpg --mode mosaic --strength 35 --margin 15
python auto_censor.py a.jpg b.png --mode solid --color ff0000
python auto_censor.py a.jpg --mode img --stamp 遮挡图.png
python auto_censor.py --list-classes   # 查看全部可打码类别
```

## 输出格式说明

- 图片：jpg（可调质量）/ png
- 视频：默认 **webm**（浏览器可直接预览，无音频）；本机安装 **ffmpeg** 后自动升级为 **H.264 mp4 + 音频**。设置菜单里可强制指定 mp4（无 ffmpeg 时使用 mp4v 编码，兼容播放器但浏览器通常无法预览）

## 自测

```bash
python selftest.py
```

覆盖参数校验、三种打码渲染、图片遮挡、检测器边界、HTTP 接口、GIF / 视频任务、输出格式等（当前 73 项断言）。

> 想给本项目贡献代码 / 接手开发？请看 [DEVELOPMENT.md](DEVELOPMENT.md)：架构说明、HTTP 接口、安全约定与发布流程。

## 项目结构

```
webui.py        网页服务（标准库 http.server，无框架）
censor_core.py  检测 + 打码渲染核心（被 web / CLI 共用）
media_core.py   GIF / 视频逐帧处理与编码
auto_censor.py  命令行入口
check_deps.py   依赖检测与自动安装
selftest.py     自动化自测
启动打码工作台.bat  Windows 一键启动（依赖检查 + 起服务 + 开浏览器）
```

## 隐私与合规

- 所有推理、编码均在本地完成，无任何上传
- 请仅用于**你自己拥有或已获授权**的内容；请遵守当地法律法规

## 致谢

- 检测模型：[NudeNet](https://github.com/notAI-tech/NudeNet)（notAI-tech），AGPL-3.0
- 因此本项目以 **AGPL-3.0** 协议开源，详见 [LICENSE](LICENSE)

## Star History

如果这个工具对你有帮助，欢迎点个 Star ⭐
