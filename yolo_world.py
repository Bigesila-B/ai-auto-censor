# YOLO-World 开放词汇检测：任意自定义英文类别的检测与文本嵌入编码
# 两阶段 prompt-then-detect：CLIP 文本塔编码类别词 -> 检测器按嵌入检测。
# 仅在用户使用自定义类别时才加载模型/下载权重；未使用时零开销。
# 路径安全说明：所有文件路径均为代码内常量，用 os.path/join 拼出；
# 用户输入的类别词只进入模型输入与缓存 key，不进入任何文件系统路径。
import os
import threading
import urllib.request

import cv2
import numpy as np
import onnxruntime

from censor_core import SLICE_RATIO, SLICE_MIN_SIDE, _host_allowed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 固定常量目录名（代码内字面量），模型与缓存都放这里
YOLO_DIR = os.path.join(BASE_DIR, "yolo_world")
DETECTOR_PATH = os.path.join(YOLO_DIR, "yolov8s-worldv2.onnx")
TEXT_MODEL_PATH = os.path.join(YOLO_DIR, "text_model_quantized.onnx")
TOKENIZER_PATH = os.path.join(YOLO_DIR, "tokenizer.json")
EMBEDS_PATH = os.path.join(YOLO_DIR, "embeds.npz")

# 下载源（按顺序尝试）：hf-mirror 镜像优先（国内可用），其次 HuggingFace 主站。
# 全部经 _host_allowed 校验（防 SSRF），URL 为代码内常量。
DETECTOR_SOURCES = [
    "https://hf-mirror.com/Instemic/yolo-world-onnx/resolve/main/yolov8s-worldv2.onnx",
    "https://huggingface.co/Instemic/yolo-world-onnx/resolve/main/yolov8s-worldv2.onnx",
]
TEXT_MODEL_SOURCES = [
    "https://hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx",
]
TOKENIZER_SOURCES = [
    "https://hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
]

# fp32 文本塔体积提示（下载进度打印用）
TEXT_MODEL_MB = 242

# 自定义类别词约束（安全与效果双重考虑）
MAX_WORLD_CLASSES = 16      # 最多类别数
MAX_WORD_LEN = 64           # 单词最大长度
WORLD_PREFIX = "WORLD:"     # 与 NudeNet 18 类隔离的 class 标识前缀

# CLIP BPE 常量（OpenAI CLIP ViT-B/32 词表约定）
CTX_LEN = 77                # 上下文长度
BOS_ID = 49406              # <start_of_text>
EOS_ID = 49407              # <end_of_text>

# 同义词表：部分词的复数/变体在模型里响应更好（实测 foot=0.17 vs feet=0.33）。
# 用户输入的词会同时保留原词与同义词，任一命中即打码。
SYNONYMS = {
    "foot": "feet",
    "hand": "hands",
    "toe": "toes",
    "ear": "ears",
    "eye": "eyes",
    "breast": "breasts",
    "butt": "buttocks",
    "genital": "genitalia",
}


def expand_words(words):
    """扩展用户词表：每个词附加其同义词（去重、保持顺序）。"""
    out = list(words)
    for w in words:
        syn = SYNONYMS.get(w)
        if syn and syn not in out:
            out.append(syn)
    return out


# 背景类：Ultralytics 官方建议类别列表尾部追加空串作背景锚。
# 实测（zidane/bus 多图验证）：没有它，部分词（如 face）单独检测时 sigmoid
# 分数整体为 0；追加后恢复正常（face 0.0 -> 0.29）。注意：bg 通道会压低
# person 等词的分数（0.9 -> 0.2~0.4，依组合而定），因此世界模型的置信度
# 阈值与 NudeNet 滑块解耦（见 censor_core.build_cfg 的 world_conf）。
BG_CLASS = ""


def with_bg(words):
    """类别词表尾部追加背景类，返回 (扩展词表, 背景通道下标)。"""
    return list(words) + [BG_CLASS], len(words)


# 检测器输入分辨率（与 NudeNet 默认一致，CPU 速度可接受）
WORLD_RESOLUTION = 640


def normalize_world_classes(raw):
    """把用户输入归一化为合法类别词列表。

    raw: None / str（逗号分隔）/ str 列表。
    规则：trim、去空、去重（保持顺序）、小写、限长。
    超限（数量/长度）截断而不是报错。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        words = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        words = list(raw)
    else:
        return []
    out, seen = [], set()
    for w in words:
        if not isinstance(w, str):
            continue
        w = w.strip().lower()
        if not w or w in seen:
            continue
        w = w[:MAX_WORD_LEN]
        seen.add(w)
        out.append(w)
        if len(out) >= MAX_WORLD_CLASSES:
            break
    return out


def _download(sources, dest):
    """按源顺序下载文件到 dest（.part 临时名，成功后原子替换）。"""
    os.makedirs(YOLO_DIR, exist_ok=True)
    for src in sources:
        if not _host_allowed(src):
            continue
        try:
            urllib.request.urlretrieve(src, dest + ".part")
            os.replace(dest + ".part", dest)
            return
        except Exception:
            continue
    raise RuntimeError(
        "YOLO-World 模型下载失败，请检查网络后重试（也可手动下载放入 yolo_world/ 目录）")


def ensure_models():
    """确保检测器与 tokenizer/文本塔就位；缺失时自动下载。"""
    if not os.path.exists(DETECTOR_PATH):
        print("未找到 yolov8s-worldv2.onnx，开始自动下载（约 48.8MB）…")
        _download(DETECTOR_SOURCES, DETECTOR_PATH)
        print("yolov8s-worldv2.onnx 下载完成")
    need_text = (not os.path.exists(TOKENIZER_PATH)
                 or not os.path.exists(TEXT_MODEL_PATH))
    if need_text:
        print(f"未找到 CLIP 文本塔/tokenizer，开始自动下载（约 {TEXT_MODEL_MB}MB，仅首次）…")
        _download(TOKENIZER_SOURCES, TOKENIZER_PATH)
        _download(TEXT_MODEL_SOURCES, TEXT_MODEL_PATH)
        print("CLIP 文本塔/tokenizer 下载完成")


class TextEncoder:
    """CLIP ViT-B/32 文本塔：类别词 -> [K,512] L2 归一化嵌入。

    仅在嵌入缓存未命中时才会实例化（省内存与启动时间）。
    """

    def __init__(self):
        if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(TEXT_MODEL_PATH):
            ensure_models()
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError("缺少 tokenizers 库，请执行：pip install tokenizers")
        self.tok = Tokenizer.from_file(TOKENIZER_PATH)
        self.session = onnxruntime.InferenceSession(
            TEXT_MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _tokenize(self, words):
        """CLIP BPE。tokenizer.json 自带 post-processor（BOS/EOS），
        这里只做防御性截断 + EOS pad 到 77（pad token = EOS 是 CLIP 约定）。
        返回 int64 [K,77]。"""
        ids = np.full((len(words), CTX_LEN), EOS_ID, dtype=np.int64)
        encs = self.tok.encode_batch(words)
        for i, enc in enumerate(encs):
            seq = enc.ids[:CTX_LEN - 1]   # 至少保留一个 EOS
            ids[i, :len(seq)] = seq
        return ids

    def encode(self, words):
        """words: 非空英文字符串列表。返回 float32 [K,512]，每行 L2 归一化。"""
        ids = self._tokenize(words)
        out = self.session.run(None, {self.input_name: ids})[0]
        emb = np.asarray(out, dtype=np.float32)
        if emb.ndim == 3:                        # 兼容带序列维的输出
            emb = emb[:, 0, :] if emb.shape[1] == 1 else emb.squeeze(1)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        return emb


def _cache_key(words):
    """缓存 key：类别词列表的有序 JSON（顺序参与 key）。"""
    import json
    return json.dumps(list(words), ensure_ascii=False, separators=(",", ":"))


class _EmbedCache:
    """嵌入缓存：npz 内 key=类名元组 JSON，value=[K,512]。线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        if os.path.exists(EMBEDS_PATH):
            try:
                with np.load(EMBEDS_PATH) as z:
                    for k in z.files:
                        self._data[k] = z[k]
            except Exception:
                self._data = {}

    def get(self, words):
        with self._lock:
            return self._data.get(_cache_key(words))

    def put(self, words, emb):
        with self._lock:
            self._data[_cache_key(words)] = emb
            try:
                os.makedirs(YOLO_DIR, exist_ok=True)
                tmp = EMBEDS_PATH + ".tmp.npz"
                np.savez(tmp, **self._data)
                os.replace(tmp, EMBEDS_PATH)
            except Exception:
                pass  # 缓存写盘失败不影响功能（下次重新编码）


class WorldDetector:
    """YOLO-World-S 检测器。进程内常驻，线程安全（推理锁串行化）。"""

    def __init__(self):
        if not os.path.exists(DETECTOR_PATH):
            ensure_models()
        self.session = onnxruntime.InferenceSession(
            DETECTOR_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.embed_cache = _EmbedCache()
        self._text_encoder = None               # 懒创建
        self._infer_lock = threading.Lock()

    def get_embeds(self, words):
        """words: 归一化类别词列表。返回扩展词表（含背景类）的 [K+1,512] 嵌入；
        命中缓存零文本开销。缓存 key 只含真实词（背景类是固定附加项）。"""
        ext, _nbg = with_bg(words)
        emb = self.embed_cache.get(words)
        if emb is not None and emb.shape[0] == len(ext):
            return emb
        if self._text_encoder is None:
            self._text_encoder = TextEncoder()
        emb = self._text_encoder.encode(ext)
        self.embed_cache.put(words, emb)
        return emb

    def detect(self, image, words, embeds=None, conf=0.25, iou=0.45):
        """检测自定义类别。输出格式与 censor_core.Detector.detect 一致：
        [{'class': 'WORLD:gun', 'score': 0.6, 'box': [x,y,w,h]}, ...]

        words: 归一化类别词列表；embeds: 可传入 get_embeds 的结果（多帧复用）。
        长图（长宽比 > SLICE_RATIO 且短边 >= SLICE_MIN_SIDE）自动切片。
        """
        if not words:
            return []
        if embeds is None:
            embeds = self.get_embeds(words)
        h, w = image.shape[:2]
        long_ratio = max(w / h, h / w)
        if long_ratio > SLICE_RATIO and min(w, h) >= SLICE_MIN_SIDE:
            return self._detect_sliced(image, words, embeds, conf, iou)
        blob, scale, pad_w, pad_h = self._preprocess(image)
        return self._run(blob, words, embeds, conf, iou, scale, pad_w, pad_h,
                         ox=0, oy=0, img_w=w, img_h=h)

    def _preprocess(self, mat):
        """letterbox 到 WORLD_RESOLUTION：RGB、/255、保持长宽比、灰色填充。"""
        h, w = mat.shape[:2]
        if mat.ndim == 2:
            mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        scale = min(WORLD_RESOLUTION / h, WORLD_RESOLUTION / w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w = WORLD_RESOLUTION - nw
        pad_h = WORLD_RESOLUTION - nh
        canvas = np.full((WORLD_RESOLUTION, WORLD_RESOLUTION, 3), 114,
                         dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]   # [1,3,H,W]
        return blob, scale, pad_w, pad_h

    def _run(self, blob, words, embeds, conf, iou, scale, pad_w, pad_h,
             ox, oy, img_w, img_h):
        """单块（或全图）推理 + 后处理。坐标映射回 (ox,oy) 偏移的原图区域。

        embeds 为扩展词表（含背景类）的嵌入；背景通道（下标 len(words)）
        只作激活锚，不产出检测框。
        """
        txt = embeds[np.newaxis].astype(np.float32)  # [1,K+1,512]
        with self._infer_lock:
            outputs = self.session.run(
                None, {self.input_name: blob, "txt_feats": txt})
        preds = np.squeeze(outputs[0], axis=0).T     # [8400, 4+K+1]
        ncls = len(words)
        boxes, scores, class_ids = [], [], []
        for row in preds:
            cs = row[4:4 + ncls]                     # 已 sigmoid
            m = float(cs.max())
            if m < conf:
                continue
            cx, cy, bw, bh = row[:4]                 # 输入图尺度 xywh
            # letterbox 逆映射（填充在右/下侧，无偏移项）
            x = cx / scale
            y = cy / scale
            bw = bw / scale
            bh = bh / scale
            x = max(0.0, min(x, img_w))
            y = max(0.0, min(y, img_h))
            bw = min(bw, img_w - x)
            bh = min(bh, img_h - y)
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x + ox, y + oy, bw, bh])
            scores.append(m)
            class_ids.append(int(cs.argmax()))
        if not boxes:
            return []
        indices = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf, iou)).flatten()
        dets = []
        for i in indices:
            x, y, bw, bh = boxes[i]
            dets.append({
                "class": WORLD_PREFIX + words[class_ids[i]],
                "score": float(scores[i]),
                "box": [int(x), int(y), int(bw), int(bh)],
            })
        return dets

    def _detect_sliced(self, image, words, embeds, conf, iou):
        """长图切片检测：与 censor_core.Detector._detect_sliced 同规则。"""
        h, w = image.shape[:2]
        tile = min(max(min(h, w), 480), 960)
        overlap = int(tile * 0.2)
        step = tile - overlap
        dets = []
        y0 = 0
        while y0 < h:
            y1 = min(y0 + tile, h)
            patch = image[y0:y1, :, :]
            if patch.shape[0] < 64:
                break
            blob, scale, pad_w, pad_h = self._preprocess(patch)
            dets.extend(self._run(blob, words, embeds, conf, iou, scale,
                                  pad_w, pad_h, ox=0, oy=y0,
                                  img_w=w, img_h=patch.shape[0]))
            y0 += step
        if not dets:
            return []
        # 跨块 NMS 去重（同类别间）
        boxes = [d["box"] for d in dets]
        scores = [d["score"] for d in dets]
        wids = [0 if d["class"].startswith(WORLD_PREFIX) else 1 for d in dets]
        words_lower = [d["class"] for d in dets]
        idx_map = {c: i for i, c in enumerate(sorted(set(words_lower)))}
        class_ids = [idx_map[c] for c in words_lower]
        # 按类别分组做 NMS（cv2 NMSBoxes 不区分类别）
        keep = []
        for cid in set(class_ids):
            group = [i for i, c in enumerate(class_ids) if c == cid]
            gb = [ [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]] for i in group ]
            gs = [scores[i] for i in group]
            gi = np.array(cv2.dnn.NMSBoxes(gb, gs, conf, iou)).flatten()
            keep.extend(group[i] for i in gi)
        return [dets[i] for i in sorted(keep)]


# ---------- 进程内单例（懒加载，供 webui / CLI 使用） ----------

_WORLD_DETECTOR = None
_WORLD_LOCK = threading.Lock()


def get_world_detector():
    """懒加载全局 WorldDetector 单例。首次调用可能触发模型下载。"""
    global _WORLD_DETECTOR
    if _WORLD_DETECTOR is None:
        with _WORLD_LOCK:
            if _WORLD_DETECTOR is None:
                _WORLD_DETECTOR = WorldDetector()
    return _WORLD_DETECTOR
