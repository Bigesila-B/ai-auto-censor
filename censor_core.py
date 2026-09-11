# 共享核心：模型加载、检测（可调阈值/推理分辨率）、多种打码渲染
# 被 webui.py 和 auto_censor.py 共用
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 固定常量名，直接拼接（与 os.path.join 等价，且均为代码内字面量）
MODEL_640M = BASE_DIR + os.sep + "640m.onnx"
# 回退模型（主模型下载失败时使用，发布版 Release 附件亦提供）
MODEL_320N = BASE_DIR + os.sep + "320n.onnx"
# 模型下载源（按顺序尝试）：本仓库 Release 附件优先，其次 hf-mirror 镜像
MODEL_SOURCES = [
    "https://github.com/Bigesila-B/ai-auto-censor/releases/latest/download/640m.onnx",
    "https://hf-mirror.com/zhangsongbo365/nudenet_onnx/resolve/main/640m.onnx",
]

LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# 默认只打码隐私部位；脸/四肢等其他类别由用户自行勾选
DEFAULT_CLASSES = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]

CENSOR_MODES = ("mosaic", "blur", "solid", "img")
ALLOWED_RESOLUTIONS = (320, 640, 960, 1280)
# 长图自动切片：长宽比超过该值、且短边不小于该像素时启用切片检测
SLICE_RATIO = 2.5
SLICE_MIN_SIDE = 480


def _host_allowed(url):
    """仅允许 http/https 且 host 不是本机/私网/保留地址（防 SSRF）。"""
    try:
        from urllib.parse import urlparse
        import ipaddress
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        host = p.hostname.strip("[]")
        if host in ("localhost",) or host.endswith(".localhost"):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        except ValueError:
            pass  # 域名（非 IP）交由 DNS 解析后由系统栈处理
        return True
    except Exception:
        return False


def ensure_model():
    """优先使用 640m 权重；缺失时按 MODEL_SOURCES 顺序自动下载，全失败回退 320n。"""
    if os.path.exists(MODEL_640M):
        return MODEL_640M
    print("未找到 640m.onnx，开始自动下载（约 100MB）…")
    for src in MODEL_SOURCES:
        if not _host_allowed(src):
            continue
        tag = "本仓库 Release" if "Bigesila-B" in src else "hf-mirror 镜像"
        print(f"  尝试下载源：{tag} …")
        try:
            urllib.request.urlretrieve(src, MODEL_640M + ".part")
            os.replace(MODEL_640M + ".part", MODEL_640M)
            print(f"640m.onnx 下载完成（来自{tag}）")
            return MODEL_640M
        except Exception as e:
            print(f"  该源下载失败（{e}）")
    print("全部下载源失败，回退到 320n 模型（可从 Release 页手动下载 640m.onnx 放到程序目录）")
    return MODEL_320N


class Detector:
    """NudeNet ONNX 检测器，支持自定义置信度阈值与推理分辨率。

    预处理与上游 NudeNet 发布代码保持一致（cvtColor(RGBA2BGR) + swapRB=True），
    官方权重即按这条管线验证，请勿单独"修正"通道顺序。
    """

    def __init__(self, model_path=None, inference_resolution=640):
        path = model_path or ensure_model()
        self.session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.resolution = inference_resolution

    def detect(self, image, conf=0.25, iou=0.45):
        """image: BGR ndarray。返回 [{'class','score','box':[x,y,w,h]}, ...]

        长图（长宽比 > SLICE_RATIO）自动切片检测：按固定块尺寸切成若干
        重叠小块，各自送模型，再把检测框映射回原图坐标并用 NMS 去重。
        """
        h, w = image.shape[:2]
        long_ratio = max(w / h, h / w)
        if long_ratio > SLICE_RATIO and min(w, h) >= SLICE_MIN_SIDE:
            return self._detect_sliced(image, conf, iou)
        blob, padded = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, padded, image, conf, iou)

    def _detect_sliced(self, image, conf, iou):
        """长图切片检测：小块 + 重叠 + 映射回原坐标 + NMS 去重。"""
        h, w = image.shape[:2]
        # 块尺寸取短边（长图短边通常够宽），限制在 480~960
        tile = min(max(min(h, w), 480), 960)
        overlap = int(tile * 0.2)   # 20% 重叠，避免目标被切在边界
        step = tile - overlap
        boxes, scores, class_ids = [], [], []
        y0 = 0
        while y0 < h:
            y1 = min(y0 + tile, h)
            patch = image[y0:y1, :, :]
            if patch.shape[0] < 64:   # 尾部残片太小直接跳过
                break
            blob, padded = self._preprocess(patch)
            outputs = self.session.run(None, {self.input_name: blob})
            preds = np.transpose(np.squeeze(outputs[0]))
            scale = padded / self.resolution
            ph = patch.shape[0]
            for row in preds:
                cs = row[4:]
                ms = float(cs.max())
                if ms < conf:
                    continue
                cx, cy, pw, phh = row[:4]
                x = (cx - pw / 2) * scale
                y = (cy - phh / 2) * scale
                x = max(0.0, min(x, w))
                y = max(0.0, min(y, ph))
                pw = min(pw * scale, w - x)
                phh = min(phh * scale, ph - y)
                if pw <= 0 or phh <= 0:
                    continue
                # 映射回原图坐标
                boxes.append([x, y + y0, pw, phh])
                scores.append(ms)
                class_ids.append(int(cs.argmax()))
            y0 += step
        if not boxes:
            return []
        indices = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf, iou)).flatten()
        detections = []
        for i in indices:
            x, y, bw, bh = boxes[i]
            detections.append({
                "class": LABELS[class_ids[i]],
                "score": float(scores[i]),
                "box": [int(x), int(y), int(bw), int(bh)],
            })
        return detections

    def _preprocess(self, mat):
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
        max_size = max(mat_c3.shape[:2])
        mat_pad = cv2.copyMakeBorder(
            mat_c3, 0, max_size - mat_c3.shape[0],
            0, max_size - mat_c3.shape[1], cv2.BORDER_CONSTANT)
        blob = cv2.dnn.blobFromImage(
            mat_pad, 1 / 255.0, (self.resolution, self.resolution),
            (0, 0, 0), swapRB=True, crop=False)
        return blob, max_size

    def _postprocess(self, outputs, padded, image, conf, iou):
        preds = np.transpose(np.squeeze(outputs[0]))
        oh, ow = image.shape[:2]
        scale = padded / self.resolution

        boxes, scores, class_ids = [], [], []
        for row in preds:
            class_scores = row[4:]
            max_score = float(class_scores.max())
            if max_score < conf:
                continue
            cx, cy, w, h = row[:4]
            x = (cx - w / 2) * scale
            y = (cy - h / 2) * scale
            w = w * scale
            h = h * scale
            x = max(0.0, min(x, ow))
            y = max(0.0, min(y, oh))
            w = min(w, ow - x)
            h = min(h, oh - y)
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, w, h])
            scores.append(max_score)
            class_ids.append(int(class_scores.argmax()))

        if not boxes:
            return []
        indices = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf, iou)).flatten()
        detections = []
        for i in indices:
            x, y, w, h = boxes[i]
            detections.append({
                "class": LABELS[class_ids[i]],
                "score": float(scores[i]),
                "box": [int(x), int(y), int(w), int(h)],
            })
        return detections


def parse_hex_color(color):
    """'#rrggbb' / 'rrggbb' -> BGR tuple；非法输入返回黑色。"""
    if not isinstance(color, str):
        return (0, 0, 0)
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (0, 0, 0)
    try:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)
    return (b, g, r)


def build_cfg(mode="mosaic", strength=35, margin=15, color="#000000",
              classes=None, conf=0.25, asset=None, world_classes=None):
    """校验并归一化打码配置，webui 与 CLI 共用。

    classes: None 用默认隐私部位集合；["all"] 表示全部类别。
    asset: 图片遮挡模式的遮挡图资源 id（服务端 /asset 返回的 hex）。
    world_classes: YOLO-World 自定义类别词列表（已由
        yolo_world.normalize_world_classes 归一化），None 表示不启用。
    """
    if mode not in CENSOR_MODES:
        mode = "mosaic"
    try:
        strength = int(strength)
    except (TypeError, ValueError):
        strength = 35
    strength = max(1, min(100, strength))
    try:
        margin = int(margin)
    except (TypeError, ValueError):
        margin = 15
    margin = max(0, min(60, margin))
    if classes is None:
        classes = list(DEFAULT_CLASSES)
    elif classes == ["all"]:
        classes = list(LABELS)
    classes = [c for c in classes if c in LABELS]
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.25
    conf = max(0.04, min(0.9, conf))
    asset = str(asset) if asset is not None else None
    if asset is not None and not (asset.isalnum() and len(asset) <= 64):
        asset = None
    if world_classes in (None, "", []):
        world_classes = []
    else:
        # 支持字符串（逗号分隔）或列表；归一化交给 yolo_world（延迟导入避免循环依赖）
        from yolo_world import normalize_world_classes
        world_classes = normalize_world_classes(world_classes)
    return {
        "mode": mode,
        "strength": strength,
        "margin": margin,
        "color_bgr": parse_hex_color(color),
        "classes": classes,
        "conf": conf,
        "asset": asset,
        "world_classes": world_classes,
    }


def censor_regions(img, detections, cfg, stamp=None):
    """按 cfg 在 img 上就地打码，返回实际处理的区域数。

    stamp: mode="img" 时使用的遮挡图（BGR ndarray），按检测框等比例放大、
    居中裁剪至完全覆盖；缺失时该模式退化为纯色填充。
    """
    scale = max(img.shape[:2]) / 640.0
    margin = cfg["margin"] / 100.0
    mode, color = cfg["mode"], cfg["color_bgr"]
    classes = cfg["classes"]
    h_img, w_img = img.shape[:2]
    count = 0
    for d in detections:
        if d["class"] not in classes:
            continue
        x, y, w, h = d["box"]
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w_img, x + w + mx), min(h_img, y + h + my)
        if x1 <= x0 or y1 <= y0:
            continue
        bw, bh = x1 - x0, y1 - y0
        if mode == "solid":
            img[y0:y1, x0:x1] = color
        elif mode == "img":
            if stamp is not None and stamp.size:
                ih, iw = stamp.shape[:2]
                scale = max(bw / iw, bh / ih)  # 等比例放大至完全覆盖
                nw = max(bw, int(round(iw * scale)))
                nh = max(bh, int(round(ih * scale)))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                big = cv2.resize(stamp, (nw, nh), interpolation=interp)
                cx = (nw - bw) // 2
                cy = (nh - bh) // 2
                img[y0:y1, x0:x1] = big[cy:cy + bh, cx:cx + bw]
            else:
                img[y0:y1, x0:x1] = color  # 未提供遮挡图片时退化为纯色
        elif mode == "mosaic":
            # 强度 1 -> 约 2px 细块，35 -> 约 24px，100 -> 64px（按 640 图折算，随图片尺寸缩放）
            block = max(2, round((2 + 0.62 * cfg["strength"]) * scale))
            small = cv2.resize(img[y0:y1, x0:x1],
                               (max(1, bw // block), max(1, bh // block)),
                               interpolation=cv2.INTER_LINEAR)
            img[y0:y1, x0:x1] = cv2.resize(small, (bw, bh),
                                           interpolation=cv2.INTER_NEAREST)
        elif mode == "blur":
            k = int(round((3 + 0.6 * cfg["strength"]) * scale))
            k = min(k, bw, bh)
            if k >= 3:
                if k % 2 == 0:
                    k -= 1
                img[y0:y1, x0:x1] = cv2.GaussianBlur(
                    img[y0:y1, x0:x1], (k, k), 0)
            else:
                img[y0:y1, x0:x1] = color  # 区域太小无法模糊，退化为纯色
        count += 1
    return count
