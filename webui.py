# 本地网页版自动打码服务（仅使用标准库 + onnxruntime + opencv + Pillow，无新增依赖）
# 启动: python webui.py [端口]  然后浏览器打开 http://localhost:8080
# 模型: 默认 640m（缺失时自动从 hf-mirror 下载，失败回退 320n）
# 前端: AI 模式（批量图片/GIF/视频，手动点击开始）/ 手动模式（画笔编辑器，可叠加 AI 预打码）
import base64
import json
import os
import socket
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from censor_core import (Detector, LABELS, DEFAULT_CLASSES,
                         ALLOWED_RESOLUTIONS, build_cfg, censor_regions, ensure_model)
from media_core import process_gif, process_video, _find_ffmpeg
from yolo_world import (normalize_world_classes, expand_words,
                        get_world_detector, WORLD_PREFIX)

ensure_model()
detector = Detector()  # 640m @ 640，每次请求可按参数切换推理分辨率
HAS_FFMPEG = _find_ffmpeg() is not None


def _detect_combined(img, cfg):
    """NudeNet + YOLO-World（可选）合并检测。

    cfg["world_classes"] 为空时与原 detector.detect 行为完全一致（零开销）。
    非空时附加 YOLO-World 检测（同义词扩展 + world_conf 独立阈值）；
    模型缺失/下载失败抛 RuntimeError（由调用方转为任务失败并提示用户）。
    返回 (合并检测结果, 启用的 WORLD 类别标识列表)。
    """
    detector.resolution = cfg["res"]
    detections = detector.detect(img, conf=cfg["conf"])
    world_words = cfg.get("world_classes") or []
    if not world_words:
        return detections, []
    all_words = expand_words(world_words)
    wd = get_world_detector()
    embeds = wd.get_embeds(all_words)
    world_dets = wd.detect(img, all_words, embeds=embeds,
                           conf=cfg.get("world_conf", 0.15))
    enabled = [WORLD_PREFIX + w for w in all_words]
    return detections + world_dets, enabled

# ---------------- 媒体任务（GIF/视频） ----------------
MEDIA_ROOT = Path(tempfile.mkdtemp(prefix="censor_media_"))
MEDIA_LIMIT = 500 * 1024 * 1024   # 单个上传文件上限
MEDIA_KEEP = 6                    # 最多保留的已完成任务结果数
MEDIA_LOCK = threading.Lock()
MEDIA = {}                        # id -> job dict
MEDIA_MIME = {"gif": "image/gif", "webm": "video/webm", "mp4": "video/mp4"}
CURRENT_PORT = 8080   # 运行时由 __main__ 更新

# ---------------- 局域网访问 ----------------
LAN_HOST = "0.0.0.0"
LOOPBACK_HOST = "127.0.0.1"
CURRENT_LAN = False   # 运行时由 __main__ 更新：是否允许局域网访问


def _lan_ips():
    """返回本机局域网 IP 列表（尽力而为，优先真实局域网段）。"""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    # 补充 UDP connect 法拿到的出口 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    # 去重并排序：优先 192.168/10./172.16-31 真实局域网段，其次其他
    uniq = sorted(set(ips))
    def rank(ip):
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if ip.startswith("172."):
            try:
                o = int(ip.split(".")[1])
                if 16 <= o <= 31:
                    return 2
            except Exception:
                pass
        return 3
    return sorted(uniq, key=rank)


def _restart_with_mode(mode):
    """LAN 模式切换：用 os.execv 替换自身进程（同端口、不同绑定），
    旧进程随即退出，避免新旧进程争抢端口。"""
    import os as _os
    python = sys.executable
    args = [python, _os.path.abspath(__file__), str(CURRENT_PORT)]
    if mode == "lan":
        args.append("--lan")
    _os.execv(python, args)   # 替换当前进程，不会返回


# ---------------- 遮挡图片素材（图片打码模式） ----------------
ASSET_LIMIT = 8 * 1024 * 1024     # 单张遮挡图上限
ASSET_KEEP = 8                    # 最多缓存的素材数
ASSETS = {}                       # id -> {"img": ndarray, "ts": float}


def _media_worker(job_id, data, kind, cfg, stamp):
    job = MEDIA[job_id]

    def cb(done, total):
        with MEDIA_LOCK:
            job["frame"], job["frames"] = done, total
            job["pct"] = min(99, int(done * 100 / max(1, total)))

    try:
        # YOLO-World 懒加载放在任务线程里做（下载可能耗时，避免卡 HTTP 响应）
        world_words = expand_words(cfg.get("world_classes") or [])
        world_detector = get_world_detector() if world_words else None
        world_embeds = (world_detector.get_embeds(world_words)
                        if world_detector else None)
        enabled_world = [WORLD_PREFIX + w for w in world_words]
        if enabled_world and cfg["classes"] is not None:
            cfg["classes"] = list(cfg["classes"]) + enabled_world
        cfg["world_classes"] = world_words
        detector.resolution = cfg["res"]
        if kind == "gif":
            out, ext, info = process_gif(data, cfg, detector, stamp=stamp,
                                         progress=cb, world=world_detector,
                                         world_embeds=world_embeds)
        else:
            out, ext, info = process_video(data, cfg, detector, stamp=stamp,
                                           progress=cb, prefer=cfg["vfmt"],
                                           world=world_detector,
                                           world_embeds=world_embeds)
        # job_id 是 uuid hex，ext 只可能来自 {"gif","webm","mp4"}
        path = MEDIA_ROOT / (job_id + "." + ext)
        path.write_bytes(out)
        with MEDIA_LOCK:
            job.update(st="done", pct=100, path=str(path), ext=ext, info=info)
    except Exception as e:
        with MEDIA_LOCK:
            job.update(st="fail", err=str(e))
    finally:
        with MEDIA_LOCK:  # 只保留最近 MEDIA_KEEP 个已产出结果的任务
            finished = sorted((j for j in MEDIA.values() if j.get("path")),
                              key=lambda j: j["ts"])
            for old in finished[:-MEDIA_KEEP]:
                try:
                    os.remove(old["path"])
                except OSError:
                    pass
                MEDIA.pop(old["id"], None)


def _media_snapshot(job):
    """读取任务快照；调用方必须已持有 MEDIA_LOCK（Lock 不可重入）。"""
    return {"st": job["st"], "pct": job.get("pct", 0),
            "frame": job.get("frame"), "frames": job.get("frames"),
            "err": job.get("err"), "ext": job.get("ext"),
            "info": job.get("info")}


def _resolve_stamp(cfg):
    """图片遮挡模式下按 cfg["asset"] 取出遮挡图；其余模式返回 None。"""
    if cfg["mode"] != "img":
        return None, None
    with MEDIA_LOCK:
        entry = ASSETS.get(cfg["asset"] or "")
    if entry is None:
        return None, "请先上传遮挡图片（打码方式 → 图片遮挡）"
    return entry["img"], None

# ---------------- 页面 ----------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>打码工作台</title>
<style>
:root{
  --bg:#101318; --panel:#171b23; --panel2:#1d232e; --line:#262d3a; --line2:#323b4d;
  --text:#e7eaf0; --dim:#939cab; --amber:#d9a441; --amber-soft:rgba(217,164,65,.14);
  --ok:#58c98f; --danger:#e2606a; --mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
  --thumb:#0a0c10; --float:rgba(10,12,16,.72);
}
:root[data-theme="light"]{
  --bg:#eef0f4; --panel:#ffffff; --panel2:#e9ecf2; --line:#d8dde6; --line2:#c3cad7;
  --text:#1b202b; --dim:#5f6878; --amber:#a97a17; --amber-soft:rgba(169,122,23,.12);
  --ok:#1f9d63; --danger:#c9414e; --thumb:#dde1e9; --float:rgba(255,255,255,.85);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.55 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;
  background-image:radial-gradient(1100px 380px at 50% -160px,var(--amber-soft),transparent 70%);}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer}
img,video{display:block}
:focus-visible{outline:2px solid var(--amber);outline-offset:2px;border-radius:6px}
body[data-mode="ai"] .man-only{display:none!important}
body[data-mode="manual"] .ai-only{display:none!important}

header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:14px 22px;border-bottom:1px solid var(--line)}
.logo{width:26px;height:26px;border-radius:8px;flex:none;
  background:conic-gradient(from 210deg,var(--amber),#8a5a17 40%,var(--amber) 75%);
  box-shadow:0 0 0 1px var(--line2) inset}
h1{font-size:17px;font-weight:650;margin:0;letter-spacing:.5px}
.sub{color:var(--dim);font-size:12px;font-family:var(--mono)}
#stat{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--dim);
  background:var(--panel);border:1px solid var(--line);border-radius:99px;padding:4px 12px}
.icon-btn{width:32px;height:32px;border:1px solid var(--line);border-radius:9px;
  background:var(--panel);display:grid;place-items:center;color:var(--dim)}
.icon-btn:hover{color:var(--text);border-color:var(--line2)}

.wrap{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px 22px;max-width:1400px;margin:0 auto}
aside{position:sticky;top:16px;align-self:start;background:var(--panel);
  border:1px solid var(--line);border-radius:16px;padding:16px;max-height:calc(100vh - 40px);overflow:auto}
.sec{font-size:11px;color:var(--dim);letter-spacing:2px;margin:18px 0 8px;
  display:flex;align-items:center;gap:8px}
.sec:first-child{margin-top:0}
.sec::after{content:"";flex:1;height:1px;background:var(--line)}

.seg{display:flex;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:3px;gap:3px}
.seg button{flex:1;padding:7px 4px;border-radius:8px;color:var(--dim);font-size:13px;transition:.15s}
.seg button:hover{color:var(--text)}
.seg button.on{background:linear-gradient(160deg,#e0b258,#b9832c);color:#191204;font-weight:650;
  box-shadow:0 2px 8px rgba(217,164,65,.25)}
.seg.wrap4{flex-wrap:wrap}
.seg.wrap4 button{flex:1 1 44%}
.ctl{display:flex;align-items:center;gap:10px;margin:10px 0;font-size:13px}
.ctl label{width:58px;color:var(--dim);flex:none}
input[type=range]{flex:1;accent-color:var(--amber);height:18px}
input[type=color]{width:46px;height:30px;border:1px solid var(--line2);border-radius:8px;
  background:var(--panel2);padding:2px}
.pill{font-family:var(--mono);font-size:12px;background:var(--panel2);border:1px solid var(--line);
  border-radius:7px;padding:2px 8px;min-width:52px;text-align:center;color:var(--text)}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line2);
  border-radius:8px;padding:6px 8px;font:inherit;flex:1}
.note{font-size:11px;color:var(--dim);margin:-4px 0 8px}

#classes{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:12px;padding:4px 10px;border-radius:99px;border:1px solid var(--line);
  color:var(--dim);cursor:pointer;user-select:none;transition:.12s;background:var(--panel2)}
.chip:hover{border-color:var(--line2);color:var(--text)}
.chip.on{border-color:rgba(217,164,65,.55);background:var(--amber-soft);color:var(--amber)}
.chiprow{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.mini{font-size:12px;color:var(--dim);border:1px solid var(--line);border-radius:8px;padding:4px 12px}
.mini:hover{color:var(--text);border-color:var(--line2)}
.actions{display:flex;gap:8px;margin-top:14px}
.actions .mini{flex:1;padding:8px 0;text-align:center}
.big-btn{width:100%;padding:11px 0;border-radius:10px;font-size:14px;font-weight:650;
  background:linear-gradient(160deg,#e0b258,#b9832c);color:#191204;
  box-shadow:0 2px 10px rgba(217,164,65,.25)}
.big-btn:disabled{background:var(--panel2);color:var(--dim);box-shadow:none;cursor:not-allowed}
#stampPrev,#brushStampPrev{width:40px;height:40px;object-fit:cover;border-radius:8px;
  border:1px solid var(--line2);display:none}

main{min-width:0}
#drop{border:1.5px dashed var(--line2);border-radius:16px;background:var(--panel);
  padding:30px 20px;text-align:center;cursor:pointer;transition:.18s;position:relative;overflow:hidden}
#drop:hover{border-color:var(--dim)}
#drop.over{border-color:var(--amber);background:var(--amber-soft);
  box-shadow:0 0 34px rgba(217,164,65,.13) inset}
#drop svg{margin:0 auto 8px;opacity:.75}
#drop .big{font-size:16px;font-weight:650}
#drop .hint{color:var(--dim);font-size:12px;margin-top:4px}

#editor{margin-top:16px;background:var(--panel);border:1px solid var(--line);
  border-radius:16px;padding:14px}
.ed-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:13px}
.ed-head .tip{color:var(--dim);font-size:11px}
#edName{font-family:var(--mono);font-size:12px}
.canvas-stack{position:relative;display:inline-block;max-width:100%;line-height:0}
#edBase{max-width:100%;max-height:62vh;display:block}
#edPaint{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
#edStrip{display:flex;gap:8px;overflow-x:auto;margin-top:12px;padding-bottom:4px}
#edStrip img{height:60px;border-radius:8px;border:1px solid var(--line);cursor:pointer;flex:none}
#edStrip img.cur{outline:2px solid var(--amber)}

#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:14px;margin-top:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;position:relative}
.card.pick{cursor:pointer}
.card.pick:hover{border-color:var(--amber)}
.thumb{position:relative;aspect-ratio:1;background:var(--thumb);overflow:hidden}
.thumb img,.thumb video{width:100%;height:100%;object-fit:cover}
.thumb video{object-fit:contain;background:#000}
.st{position:absolute;top:8px;left:8px;font-family:var(--mono);font-size:11px;
  background:var(--float);border:1px solid var(--line2);border-radius:99px;padding:2px 9px;backdrop-filter:blur(3px)}
.card.done .st{color:var(--ok);border-color:var(--ok)}
.card.fail .st{color:var(--danger);border-color:var(--danger)}
.card.run .st,.card.pending .st{color:var(--amber);border-color:var(--amber)}
@keyframes scan{0%{top:-16%}100%{top:106%}}
.card.run .thumb::after{content:"";position:absolute;left:0;right:0;height:16%;
  background:linear-gradient(180deg,transparent,rgba(217,164,65,.30) 42%,rgba(217,164,65,.62) 50%,rgba(217,164,65,.30) 58%,transparent);
  animation:scan 1.15s linear infinite}
.bar{position:absolute;left:0;right:0;bottom:0;height:4px;background:var(--panel2)}
.bar i{display:block;height:100%;width:0;background:var(--amber);transition:width .4s}
.peek{position:absolute;left:8px;bottom:8px;font-size:11px;color:var(--text);
  background:var(--float);border:1px solid var(--line2);border-radius:8px;padding:3px 9px;
  opacity:0;transition:.15s}
.card:hover .peek{opacity:1}
.peek:active{border-color:var(--amber)}
.dl{position:absolute;top:8px;right:8px;display:none;width:28px;height:28px;border-radius:8px;
  background:var(--float);border:1px solid var(--line2);place-items:center;color:var(--text)}
.card.done .dl{display:grid}
.dl:hover{border-color:var(--amber)}
.cap{padding:9px 11px;border-top:1px solid var(--line)}
.name{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.subline{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:2px}
.tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.tag{font-family:var(--mono);font-size:10.5px;background:var(--panel2);border:1px solid var(--line);
  border-radius:6px;padding:1px 6px;color:var(--dim)}
.tag.hit{color:var(--amber);border-color:var(--amber)}

#empty{color:var(--dim);text-align:center;padding:44px 0;font-size:13px}
footer{color:var(--dim);font-size:11px;text-align:center;padding:18px 0 26px}

.overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:grid;place-items:center;z-index:60}
.modal{background:var(--panel);border:1px solid var(--line2);border-radius:16px;
  padding:18px;width:min(460px,92vw);max-height:86vh;overflow:auto}
.modal .head{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.modal h2{margin:0;font-size:15px}
.set-row{display:flex;align-items:center;gap:10px;margin:12px 0;font-size:13px;flex-wrap:wrap}
.set-row label.main{width:88px;color:var(--dim);flex:none}
.set-row input[type=text]{flex:1;background:var(--panel2);color:var(--text);
  border:1px solid var(--line2);border-radius:8px;padding:6px 8px;font:inherit;min-width:120px}
.set-row input[type=range]{flex:1}
.set-actions{display:flex;gap:8px;margin-top:16px}
.set-actions .mini{flex:1;text-align:center;padding:8px 0}
#worldClasses{width:100%;box-sizing:border-box;background:var(--panel2);color:var(--text);
  border:1px solid var(--line2);border-radius:8px;padding:7px 9px;font:inherit}

/* 灯箱（点击卡片全图预览） */
#lightbox{position:fixed;inset:0;background:rgba(6,8,12,.9);z-index:80;display:none;
  align-items:center;justify-content:center;flex-direction:column;gap:12px;padding:20px}
#lightbox.show{display:flex}
#lightbox img,#lightbox video{max-width:94vw;max-height:84vh;border-radius:10px;
  background:#000;box-shadow:0 20px 60px rgba(0,0,0,.6)}
#lightbox video{max-height:78vh}
#lbCap{color:var(--dim);font-size:12px;font-family:var(--mono)}
#lbClose{position:absolute;top:16px;right:20px;width:40px;height:40px;border-radius:10px;
  background:rgba(255,255,255,.08);color:var(--text);display:grid;place-items:center;font-size:20px}
#lbClose:hover{background:rgba(255,255,255,.18)}
#lightbox .lb-hint{color:var(--dim);font-size:11px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}aside{position:static;max-height:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body data-mode="ai">
<header>
  <div class="logo" aria-hidden="true"></div>
  <div><h1>打码工作台</h1></div>
  <div class="sub">NudeNet 640m · 图片不离开电脑</div>
  <span id="stat">就绪</span>
  <button type="button" class="icon-btn" id="settingsBtn" title="设置" aria-label="打开设置">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </button>
  <button type="button" class="icon-btn" id="theme" title="切换白天/夜晚模式" aria-label="切换白天/夜晚模式"></button>
</header>

<div class="wrap">
<aside>
  <div class="sec">处理模式</div>
  <div class="seg" id="runmode" role="group" aria-label="处理模式">
    <button type="button" data-run="ai" class="on">AI 模式</button>
    <button type="button" data-run="manual">手动模式</button>
  </div>
  <div class="note" id="modeNote">拖入图片 → 点"开始打码"自动检测打码</div>

  <div class="sec">打码方式</div>
  <div class="seg wrap4" id="mode" role="group" aria-label="打码方式">
    <button type="button" data-mode="mosaic" class="on">马赛克</button>
    <button type="button" data-mode="blur">高斯模糊</button>
    <button type="button" data-mode="solid">纯色填充</button>
    <button type="button" data-mode="img">图片遮挡</button>
  </div>

  <div class="sec">参数</div>
  <div class="ctl" id="colorRow" style="display:none">
    <label for="color">颜色</label><input type="color" id="color" value="#000000">
  </div>
  <div class="ctl" id="stampRow" style="display:none">
    <label>遮挡图片</label>
    <button type="button" class="mini" id="stampPick">选择图片</button>
    <img id="stampPrev" alt="">
    <input type="file" id="stampFile" accept="image/*" hidden>
  </div>
  <div class="note" id="stampNote" style="display:none">未选择遮挡图片</div>
  <div class="ctl" id="strengthRow">
    <label for="strength">强度</label>
    <input type="range" id="strength" min="1" max="100" value="35">
    <span class="pill" id="strengthVal">35</span>
  </div>
  <div class="note" id="strengthNote">数值越大，马赛克块越大</div>
  <div class="ctl">
    <label for="margin">边缘外扩</label>
    <input type="range" id="margin" min="0" max="60" value="15">
    <span class="pill" id="marginVal">15%</span>
  </div>

  <div class="sec">检测</div>
  <div class="ctl">
    <label for="conf">阈值</label>
    <input type="range" id="conf" min="0.05" max="0.9" step="0.05" value="0.25">
    <span class="pill" id="confVal">0.25</span>
  </div>
  <div class="note">越低越敏感，误检也会变多</div>
  <div class="ctl">
    <label for="res">分辨率</label>
    <select id="res">
      <option value="320">320 · 快</option>
      <option value="640" selected>640 · 均衡</option>
      <option value="960">960</option>
      <option value="1280">1280 · 准</option>
    </select>
  </div>

  <div class="sec">打码类别</div>
  <div id="classes"></div>
  <div class="chiprow">
    <button type="button" class="mini" id="allCls">全选</button>
    <button type="button" class="mini" id="defCls">默认（隐私部位）</button>
  </div>
  <div class="sec">自定义类别（YOLO-World）</div>
  <input type="text" id="worldClasses" class="tinput" spellcheck="false"
         placeholder="英文逗号分隔，如: gun, knife, face">
  <div class="note" style="margin:4px 0 10px">用 AI 检测任意目标并打码（需填英文，如 gun=枪、face=人脸）。
    单复数会自动补同义词（如 foot 自动加 feet）。首次使用会自动下载约 300MB 模型，
    之后按填写的词自动缓存。留空则不启用。自定义类别使用固定敏感度（不受上方阈值滑块影响）。</div>

  <div class="actions ai-only">
    <button type="button" class="mini" id="downloadAll">打包下载全部</button>
    <button type="button" class="mini" id="rerun">用当前设置重新处理</button>
    <button type="button" class="mini" id="clear">清空</button>
  </div>

  <div class="sec man-only">画笔</div>
  <div class="seg wrap4 man-only" id="brush" role="group" aria-label="画笔类型">
    <button type="button" data-brush="color" class="on">色彩</button>
    <button type="button" data-brush="mosaic">马赛克</button>
    <button type="button" data-brush="blur">模糊</button>
    <button type="button" data-brush="img">图片</button>
    <button type="button" data-brush="eraser">橡皮</button>
  </div>
  <div class="ctl man-only" id="brushColorRow">
    <label for="brushColor">颜色</label>
    <input type="color" id="brushColor" value="#e03434">
  </div>
  <div class="ctl man-only" id="brushStampRow" style="display:none">
    <label>遮挡图片</label>
    <button type="button" class="mini" id="brushStampPick">选择图片</button>
    <img id="brushStampPrev" alt="">
    <input type="file" id="brushStampFile" accept="image/*" hidden>
  </div>
  <div class="ctl man-only">
    <label for="brushSize">笔刷大小</label>
    <input type="range" id="brushSize" min="4" max="160" value="36">
    <span class="pill" id="brushSizeVal">36</span>
  </div>
  <div class="note man-only">橡皮只擦除手动笔迹，不会动 AI 打码层</div>
  <div class="chiprow man-only">
    <button type="button" class="mini" id="edAi">AI 预打码</button>
    <button type="button" class="mini" id="edUndo">撤销</button>
    <button type="button" class="mini" id="edClear">清除手绘</button>
    <button type="button" class="mini" id="edReset">重置</button>
    <button type="button" class="mini" id="edDl">下载结果</button>
  </div>

  <div class="actions">
    <button type="button" class="big-btn" id="start" disabled>开始打码</button>
  </div>
</aside>

<main>
  <div id="drop" role="button" tabindex="0" aria-label="拖入或点击选择图片">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#d9a441"
         stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 15V4m0 0L8 8m4-4 4 4"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>
    </svg>
    <div class="big">把图片、GIF 或视频拖进来</div>
    <div class="hint">支持批量 · 拖入后点"开始打码"才会处理 · 也可点击选择文件</div>
  </div>
  <input type="file" id="file" accept="image/*,video/*,.gif" multiple hidden>

  <div id="editor" class="man-only" style="display:none">
    <div class="ed-head">正在编辑：<span id="edName">未选择</span>
      <span class="tip">点击下方缩略图可切换图片 · 画笔会盖在 AI 预打码结果之上</span>
    </div>
    <div class="canvas-stack"><img id="edBase" alt=""><canvas id="edPaint"></canvas></div>
  </div>
  <div id="edStrip" class="man-only"></div>

  <div id="grid"></div>
  <div id="empty">队列还空着 —— 拖几张图片进来，然后点"开始打码"</div>
</main>
</div>
<div class="overlay" id="settingsOverlay" style="display:none">
  <div class="modal">
    <div class="head"><h2>设置</h2>
      <button type="button" class="icon-btn" id="setClose" title="关闭" aria-label="关闭设置">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
      </button>
    </div>
    <div class="set-row">
      <label class="main" for="setImgFmt">图片格式</label>
      <select id="setImgFmt"><option value="jpg">jpg（体积小）</option><option value="png">png（无损）</option></select>
    </div>
    <div class="set-row" id="setJpgQRow">
      <label class="main" for="setJpgQ">图片质量</label>
      <input type="range" id="setJpgQ" min="40" max="100" value="92">
      <span class="pill" id="setJpgQVal">92</span>
    </div>
    <div class="set-row">
      <label class="main" for="setVidFmt">视频格式</label>
      <select id="setVidFmt">
        <option value="webm">webm · 推荐（浏览器可预览）</option>
        <option value="mp4">mp4（兼容播放器）</option>
      </select>
    </div>
    <div class="note">未安装 ffmpeg 时 mp4 使用 mp4v 编码，浏览器通常无法预览但可下载；安装 ffmpeg 后自动变为 H.264 + 音频</div>
    <div class="set-row">
      <label class="main" for="setSuffix">文件名后缀</label>
      <input type="text" id="setSuffix" value="_censored" spellcheck="false">
    </div>
    <div class="set-row">
      <label class="main"></label>
      <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="setAutoDl"> 打码完成后自动下载</label>
    </div>
    <div class="set-row">
      <label class="main"></label>
      <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="setRemember"> 记住打码设置（下次打开自动恢复）</label>
    </div>
    <div class="set-row">
      <label class="main"></label>
      <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="setLan"> 允许局域网访问（手机/其他电脑可访问）</label>
    </div>
    <div class="set-row" id="lanInfo" style="display:none">
      <label class="main"></label>
      <span class="note" id="lanAddr" style="margin:0"></span>
    </div>
    <div class="set-actions">
      <button type="button" class="mini" id="setReset">恢复默认</button>
      <button type="button" class="mini" id="setDone">完成</button>
    </div>
  </div>
</div>
<div id="lightbox" aria-modal="true" role="dialog" aria-label="全图预览">
  <button type="button" id="lbClose" title="关闭" aria-label="关闭预览">×</button>
  <img id="lbMedia" alt="" hidden>
  <video id="lbVideo" controls playsinline hidden></video>
  <div id="lbCap"></div>
</div>
<footer>检测与打码全部在本机完成 · 队列按顺序逐张执行<span id="audioNote"></span></footer>

<script>
const ALL_CLASSES = __ALL_CLASSES__;
const DEFAULT_CLASSES = __DEFAULT_CLASSES__;
const ZH = __ZH__;
const HAS_FFMPEG = __HASFFMPEG__;
const ALLOWED_RES = ["320", "640", "960", "1280"];
const $ = id => document.getElementById(id);

/* ---------- 主题 ---------- */
const SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
let explicitTheme = false;   // 用户手动切换过主题后为 true，不再跟随系统
function applyTheme(t, persist){
  document.documentElement.dataset.theme = t;
  $("theme").innerHTML = t === "light" ? MOON : SUN;
  if (persist) try { localStorage.setItem("cb-theme", t); } catch (e) {}
}
$("theme").onclick = () => {
  explicitTheme = true;   // 手动选择后固定，不再跟随系统
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light", true);
};
let savedTheme = null;
try { savedTheme = localStorage.getItem("cb-theme"); } catch (e) {}
const sysLight = window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches;
const systemTheme = sysLight ? "light" : "dark";
applyTheme(savedTheme === "light" || savedTheme === "dark" ? savedTheme : systemTheme, false);
// 跟随系统阶段，系统切换深浅色时实时跟随
if (window.matchMedia){
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", e => {
    if (!explicitTheme) applyTheme(e.matches ? "light" : "dark", false);
  });
}
if (!HAS_FFMPEG)
  $("audioNote").textContent = " · 未找到 ffmpeg：视频输出为 WebM 静音格式，安装 imageio-ffmpeg 后自动带音频";

/* ---------- 处理模式（AI / 手动） ---------- */
let runMode = "ai";
const modeNotes = {ai:"拖入图片 → 点\u201c开始打码\u201d自动检测打码",
                   manual:"拖入或点击缩略图编辑 · 可先 AI 预打码再手动补差"};
function setRunMode(m){
  runMode = m;
  document.body.dataset.mode = m;
  for (const b of $("runmode").children) b.classList.toggle("on", b.dataset.run === m);
  $("modeNote").textContent = modeNotes[m];
  $("start").textContent = m === "ai" ? "开始打码" : "开始处理 GIF/视频";
  refreshModeUI(); refreshStart();
}
$("runmode").addEventListener("click", e => {
  const b = e.target.closest("button[data-run]");
  if (b) setRunMode(b.dataset.run);
});

/* ---------- 打码方式 ---------- */
const modeNotes2 = {mosaic:"数值越大，马赛克块越大", blur:"数值越大，模糊越重",
                    solid:"", img:"遮挡图片会等比例放大、居中盖住检测区域"};
let censorMode = "mosaic";
function refreshModeUI(){
  for (const b of $("mode").children) b.classList.toggle("on", b.dataset.mode === censorMode);
  const manual = runMode === "manual";
  const brush = $("brush").querySelector(".on").dataset.brush;
  $("colorRow").style.display = (!manual && censorMode === "solid") ? "flex" : "none";
  $("stampRow").style.display = (!manual && censorMode === "img") ? "flex" : "none";
  $("stampNote").style.display = (!manual && censorMode === "img") ? "block" : "none";
  $("strengthRow").style.display = manual
    ? (brush === "mosaic" || brush === "blur" ? "flex" : "none")
    : (censorMode === "solid" ? "none" : "flex");
  $("strengthNote").textContent = manual
    ? (brush === "mosaic" ? "数值越大，画笔马赛克块越大" : "数值越大，画笔模糊越重")
    : (modeNotes2[censorMode] || "");
  $("brushColorRow").style.display = (manual && brush === "color") ? "flex" : "none";
  $("brushStampRow").style.display = (manual && brush === "img") ? "flex" : "none";
  refreshStart();
}
$("mode").addEventListener("click", e => {
  const b = e.target.closest("button[data-mode]");
  if (b){ censorMode = b.dataset.mode; refreshModeUI(); }
});
$("brush").addEventListener("click", e => {
  const b = e.target.closest("button[data-brush]");
  if (b){ b.parentElement.querySelectorAll("button").forEach(x => x.classList.remove("on"));
         b.classList.add("on"); refreshModeUI(); }
});
const shown = (id, suffix="") => $(id + "Val").textContent = $(id).value + suffix;
$("strength").oninput = () => shown("strength");
$("margin").oninput = () => shown("margin", "%");
$("conf").oninput = () => shown("conf");
$("brushSize").oninput = () => shown("brushSize");

function chip(cls, on){
  const el = document.createElement("button");
  el.type = "button";
  el.className = "chip" + (on ? " on" : "");
  el.dataset.cls = cls; el.textContent = ZH[cls] || cls; el.title = cls;
  el.onclick = () => el.classList.toggle("on");
  return el;
}
function renderClasses(sel){
  $("classes").replaceChildren(...ALL_CLASSES.map(c => chip(c, sel.includes(c))));
}
renderClasses(DEFAULT_CLASSES);
$("allCls").onclick = () => $("classes").querySelectorAll(".chip").forEach(c => c.classList.add("on"));
$("defCls").onclick = () => renderClasses(DEFAULT_CLASSES);

function readSettings(){
  return {
    mode: censorMode,
    strength: $("strength").value, margin: $("margin").value,
    color: $("color").value, conf: $("conf").value, res: $("res").value,
    classes: [...$("classes").querySelectorAll(".chip.on")].map(c => c.dataset.cls),
    world: $("worldClasses").value.trim(),
  };
}
function paramsOf(s, assetId){
  const p = new URLSearchParams({ mode: s.mode, strength: s.strength, margin: s.margin,
    color: s.color, conf: s.conf, res: s.res, classes: s.classes.join(","),
    fmt: SET.imgFmt, quality: SET.jpgQuality, vfmt: SET.vidFmt });
  if (s.world) p.set("world", s.world);
  if (assetId) p.set("asset", assetId);
  return p;
}

/* ---------- 齿轮设置 ---------- */
const DEF_SET = { imgFmt: "jpg", jpgQuality: 92, vidFmt: "webm",
                  suffix: "_censored", autoDl: false, remember: false, ai: null };
let SET = Object.assign({}, DEF_SET);
try { Object.assign(SET, JSON.parse(localStorage.getItem("cb-settings") || "{}")); } catch (e) {}
function saveSet(){ try { localStorage.setItem("cb-settings", JSON.stringify(SET)); } catch (e) {} }
function safeSuffix(){ return (SET.suffix || "_censored").replace(/[\\\\/:*?"<>|]/g, ""); }

function syncSettingsUI(){
  $("setImgFmt").value = SET.imgFmt;
  $("setJpgQ").value = SET.jpgQuality;
  $("setJpgQVal").textContent = SET.jpgQuality;
  $("setJpgQRow").style.display = SET.imgFmt === "jpg" ? "flex" : "none";
  $("setVidFmt").value = SET.vidFmt;
  $("setSuffix").value = SET.suffix;
  $("setAutoDl").checked = SET.autoDl;
  $("setRemember").checked = SET.remember;
}
function closeSettings(){ $("settingsOverlay").style.display = "none"; }
$("setClose").onclick = closeSettings;
$("setDone").onclick = closeSettings;
$("settingsOverlay").addEventListener("click", e => {
  if (e.target === $("settingsOverlay")) closeSettings();
});
$("setImgFmt").onchange = e => { SET.imgFmt = e.target.value; saveSet(); syncSettingsUI(); };
$("setJpgQ").oninput = e => { SET.jpgQuality = +e.target.value; $("setJpgQVal").textContent = SET.jpgQuality; saveSet(); };
$("setVidFmt").onchange = e => { SET.vidFmt = e.target.value; saveSet(); };
$("setSuffix").onchange = e => { SET.suffix = e.target.value; saveSet(); };
$("setAutoDl").onchange = e => { SET.autoDl = e.target.checked; saveSet(); };
$("setRemember").onchange = e => {
  SET.remember = e.target.checked; saveSet();
  if (SET.remember){ SET.ai = readSettings(); saveSet(); }
};
$("setReset").onclick = () => {
  SET = Object.assign({}, DEF_SET, { ai: SET.remember ? DEF_SET.ai : null });
  saveSet(); syncSettingsUI();
  // 应用默认打码设置到面板
  censorMode = "mosaic";
  $("strength").value = 35; $("margin").value = 15; $("conf").value = 0.25; $("res").value = "640";
  shown("strength"); shown("margin", "%"); shown("conf");
  renderClasses(DEFAULT_CLASSES); refreshModeUI();
};

/* ---------- 局域网访问开关 ---------- */
async function refreshLanUI(){
  try {
    const j = await (await fetch("/lan")).json();
    $("setLan").checked = !!j.lan;
    const addrs = [ "http://localhost:" + j.port ];
    (j.ips || []).forEach(ip => addrs.push("http://" + ip + ":" + j.port));
    $("lanAddr").textContent = (j.lan ? "已开启，局域网设备可访问：" : "已关闭，仅本机可访问。开启后：") + addrs.join("  ·  ");
    $("lanInfo").style.display = "";
  } catch (e) { $("lanInfo").style.display = "none"; }
}
$("setLan").onchange = async e => {
  const want = e.target.checked;
  $("setLan").disabled = true;
  try {
    const j = await (await fetch("/lan?mode=" + (want ? "on" : "off"), { method: "POST" })).json();
    if (j.ok){
      if (j.restarting){
        $("lanAddr").textContent = "正在重启服务以生效…约 3 秒后请刷新页面";
        $("lanInfo").style.display = "";
        setTimeout(() => location.reload(), 3500);
      }
    } else if (j.error){ alert("切换失败：" + j.error); $("setLan").checked = !want; }
  } catch (err){ alert("切换失败：" + err); $("setLan").checked = !want; }
  $("setLan").disabled = false;
};
openSettings = () => { syncSettingsUI(); refreshLanUI(); $("settingsOverlay").style.display = "grid"; };
$("settingsBtn").onclick = openSettings;   // 绑定到带 /lan 状态刷新的新版

/* ---------- 遮挡图片（AI 图片遮挡模式） ---------- */
let stampFile = null, stampAssetId = null;
$("stampPick").onclick = () => $("stampFile").click();
$("stampFile").onchange = () => {
  const f = $("stampFile").files[0];
  if (!f || !f.type.startsWith("image/")) return;
  stampFile = f; stampAssetId = null;
  $("stampPrev").src = URL.createObjectURL(f);
  $("stampPrev").style.display = "block";
  $("stampNote").textContent = "已选择 " + f.name;
};
async function ensureStampAsset(){
  if (!stampFile) throw new Error("请先选择遮挡图片");
  if (stampAssetId) return stampAssetId;
  const resp = await fetch("/asset", { method: "POST", body: await stampFile.arrayBuffer() });
  const j = await resp.json();
  if (j.error) throw new Error(j.error);
  stampAssetId = j.id;
  return j.id;
}

/* ---------- 队列与卡片 ---------- */
const jobs = [];
let running = false;

function fmtSize(n){
  return n > 1048576 ? (n / 1048576).toFixed(1) + "MB" : Math.round(n / 1024) + "KB";
}
function fileKind(f){
  const n = f.name.toLowerCase();
  if (f.type === "image/gif" || n.endsWith(".gif")) return "gif";
  if (f.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi|m4v|mpg|mpeg|wmv|ts)$/.test(n)) return "video";
  if (f.type.startsWith("image/")) return "image";
  return null;
}
function subText(job){
  if (job.st === "fail") return job.err || "处理失败";
  const parts = [];
  if (job.dim) parts.push(job.dim[0] + "×" + job.dim[1]);
  if (job.sizeText) parts.push(job.sizeText);
  if (job.st === "done" && job.ms != null) parts.push(job.ms + "ms");
  return parts.join(" · ");
}
function statusText(job){
  if (job.st === "pending") return job.media ? "待处理" : "排队中";
  if (job.st === "run") return "处理中";
  if (job.st === "fail") return "失败";
  return job.media ? "打码完成" : `打码 ${job.res.censored_count} 处`;
}
function stat(text){
  if (text){ $("stat").textContent = text; return; }
  if (!jobs.length){ $("stat").textContent = "就绪"; return; }
  const done = jobs.filter(j => j.st === "done").length;
  const fail = jobs.filter(j => j.st === "fail").length;
  const busy = running || jobs.some(j => j.media && j.st === "run");
  $("stat").textContent = busy
    ? `处理中 ${Math.min(done + fail + 1, jobs.length)}/${jobs.length}`
    : `共 ${jobs.length} 张 · 完成 ${done} · 失败 ${fail}`;
}
function refreshStart(){
  const pending = jobs.some(j => j.st === "pending" && (j.media || runMode === "ai"));
  $("start").disabled = !pending || running;
}

function baseCard(job){
  const fig = document.createElement("figure");
  fig.className = "card " + job.st;
  fig.innerHTML = `<div class="thumb"><img alt="">
      <span class="st"></span>
      <button type="button" class="peek">按住看原图</button>
      <a class="dl" title="下载打码结果" download>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 4v12m0 0 5-5m-5 5-5-5"/><path d="M4 20h16"/></svg></a>
    </div>
    <figcaption class="cap"><div class="name"></div><div class="subline"></div><div class="tags"></div></figcaption>`;
  fig.querySelector(".name").textContent = job.file.name;
  const img = fig.querySelector("img");
  img.src = job.origUrl;
  fig.querySelector(".subline").textContent = subText(job);
  const peek = fig.querySelector(".peek");
  const showOrig = v => {
    const cur = fig.querySelector(".thumb img");
    if (cur) cur.src = (v || !job.censUrl) ? job.origUrl : job.censUrl;
  };
  peek.addEventListener("pointerdown", () => showOrig(true));
  for (const ev of ["pointerup", "pointerleave"])
    peek.addEventListener(ev, () => showOrig(false));
  return fig;
}

function paint(job){
  const el = job.el;
  el.className = "card " + job.st + (job.media ? "" : (runMode === "manual" ? " pick" : ""));
  el.querySelector(".st").textContent = statusText(job);
  el.querySelector(".subline").textContent = subText(job);
  if (job.st === "done" && !job.censUrl){
    job.censUrl = job.res.censored;
    el.querySelector("img").src = job.censUrl;
      const dl = el.querySelector(".dl");
      dl.href = job.censUrl;
      const ext = job.res.fmt || "jpg";
      dl.download = job.file.name.replace(/\\.[^.]+$/, "") + safeSuffix() + "." + ext;
      if (SET.autoDl) dl.click();
      const cls = new Set(job.s ? job.s.classes : []);
    const dets = job.res.detections;
    const tags = el.querySelector(".tags");
    tags.replaceChildren(...dets.slice(0, 4).map(t => {
      const s = document.createElement("span");
      s.className = "tag" + (cls.has(t.class) ? " hit" : "");
      s.textContent = `${ZH[t.class] || t.class} ${(t.score * 100).toFixed(0)}%`;
      return s;
    }));
    if (dets.length > 4){
      const more = document.createElement("span");
      more.className = "tag"; more.textContent = `+${dets.length - 4}`;
      tags.appendChild(more);
    }
  }
}

async function infer(file, s, assetId){
  const resp = await fetch("/infer?" + paramsOf(s, assetId),
    { method: "POST", body: await file.arrayBuffer() });
  const j = await resp.json();
  if (j.error) throw new Error(j.error);
  return j;
}

async function pump(s, assetId){
  if (running) return;
  running = true;
  try {
    while (true){
      const job = jobs.find(j => j.st === "pending" && !j.media);
      if (!job) break;
      job.s = s;
      job.st = "run"; paint(job); stat();
      const t0 = performance.now();
      try {
        job.res = await infer(job.file, s, assetId);
        job.ms = Math.round(performance.now() - t0);
        job.st = "done";
      } catch (e){ job.st = "fail"; job.err = String(e.message || e); }
      paint(job); stat();
    }
  } finally { running = false; stat(); refreshStart(); }
}

/* ---------- 媒体任务（GIF / 视频） ---------- */
function mediaCardDom(job){
  const fig = baseCard(job);
  fig.querySelector(".peek").style.display = "none";
  const bar = document.createElement("div");
  bar.className = "bar"; bar.innerHTML = "<i></i>";
  fig.querySelector(".thumb").appendChild(bar);
  return fig;
}

async function startMedia(job, s, assetId){
  try {
    const resp = await fetch("/video?" + paramsOf(s, assetId) + "&kind=" + job.kind,
      { method: "POST", body: await job.file.arrayBuffer() });
    const j = await resp.json();
    if (j.error) throw new Error(j.error);
    job.mid = j.id; job.s = s;
    job.st = "run";
    job.el.className = "card run";
    job.el.querySelector(".st").textContent = "处理中 0%";
    job.timer = setInterval(() => pollMedia(job), 600);
  } catch (e){ mediaFail(job, e); }
}

async function pollMedia(job){
  try {
    const st = await (await fetch("/video/status?id=" + job.mid)).json();
    const el = job.el;
    if (st.st === "processing"){
      el.querySelector(".bar i").style.width = (st.pct || 0) + "%";
      el.querySelector(".st").textContent = "处理中 " + (st.pct || 0) + "%";
      el.querySelector(".subline").textContent = st.frames
        ? `第 ${st.frame || 0}/${st.frames} 帧` : `第 ${st.frame || 0} 帧`;
    } else if (st.st === "done"){
      clearInterval(job.timer); job.timer = null;
      const blob = await (await fetch("/video/result?id=" + job.mid)).blob();
      job.censUrl = URL.createObjectURL(blob);
      job.st = "done";
      el.className = "card done";
      el.querySelector(".bar").remove();
      el.querySelector(".st").textContent = "打码完成";
      const ext = st.ext || (job.kind === "gif" ? "gif" : "mp4");
      if (ext === "gif"){
        el.querySelector("img").src = job.censUrl;
      } else {
        const v = document.createElement("video");
        v.src = job.censUrl; v.controls = true; v.loop = true;
        v.playsInline = true; v.preload = "metadata";
        // 不强制静音/自动播放：让用户点播放即可听到声音
        el.querySelector("img").replaceWith(v);
        el.querySelector(".peek").style.display = "none";
      }
      const dl = el.querySelector(".dl");
      dl.href = job.censUrl;
      dl.download = job.file.name.replace(/\\.[^.]+$/, "") + safeSuffix() + "." + ext;
      const info = st.info || {};
      el.querySelector(".subline").textContent = [
        job.dim ? job.dim.join("×") : null, job.sizeText,
        info.frames ? info.frames + "帧" : null,
        info.audio === false && ext !== "gif" ? "无声" : null,
      ].filter(Boolean).join(" · ");
      if (SET.autoDl) dl.click();
      stat();
    } else if (st.st === "fail"){
      clearInterval(job.timer); job.timer = null;
      mediaFail(job, new Error(st.err || "处理失败"));
    }
  } catch (e){ /* 网络/服务抖动，等下一轮 */ }
}

function mediaFail(job, e){
  if (job.timer){ clearInterval(job.timer); job.timer = null; }
  job.st = "fail"; job.err = String(e.message || e);
  job.el.className = "card fail";
  job.el.querySelector(".st").textContent = "失败";
  job.el.querySelector(".subline").textContent = job.err;
  const bar = job.el.querySelector(".bar");
  if (bar) bar.remove();
  stat();
}

/* ---------- 入队（不自动处理，等"开始打码"） ---------- */
function addFiles(list){
  const files = [...list].filter(f => fileKind(f));
  for (const f of files){
    const kind = fileKind(f);
    if (kind === "image" && runMode === "manual"){
      // 手动模式：图片只进编辑器素材条，不建 AI 队列
      loadEditor(addEdFile(f));
      continue;
    }
    if (kind === "image"){
      const job = { file: f, st: "pending",
                    origUrl: URL.createObjectURL(f), sizeText: fmtSize(f.size) };
      job.el = baseCard(job);
      paint(job);
      jobs.push(job);
      $("grid").appendChild(job.el);
      createImageBitmap(f).then(b => {
        job.dim = [b.width, b.height];
        job.el.querySelector(".subline").textContent = subText(job);
      }).catch(() => {});
    } else {
      const job = { file: f, st: "pending", media: true, kind,
                    origUrl: URL.createObjectURL(f), sizeText: fmtSize(f.size) };
      job.el = mediaCardDom(job);
      paint(job);
      jobs.push(job);
      $("grid").appendChild(job.el);
    }
  }
  if (files.length){
    $("empty").style.display = "none";
    stat(); refreshStart();
  }
}

$("start").onclick = async () => {
  $("start").disabled = true;
  try {
    const s = readSettings();
    if (SET.remember){ SET.ai = s; saveSet(); }   // 记住当前打码设置
    let assetId = null;
    if (s.mode === "img" && jobs.some(j => j.st === "pending"))
      assetId = await ensureStampAsset();
    for (const j of jobs.filter(x => x.media && x.st === "pending"))
      startMedia(j, s, assetId);
    if (runMode === "ai") await pump(s, assetId);
    else stat("手动模式：仅处理 GIF/视频，图片请在编辑器中操作");
  } catch (e){
    stat("启动失败: " + (e.message || e));
  }
  refreshStart();
};
$("rerun").onclick = async () => {
  if (running) return;
  for (const j of jobs){
    if (j.media){
      if (j.timer){ clearInterval(j.timer); j.timer = null; }
      if (j.st === "done" || j.st === "fail"){
        j.st = "pending"; j.censUrl = null; j.err = null;
        const fresh = mediaCardDom(j);
        j.el.replaceWith(fresh); j.el = fresh;
        paint(j);
      }
    } else if (j.st !== "pending"){
      j.st = "pending"; j.res = null; j.censUrl = null; j.ms = null; j.err = null;
      j.el.querySelector("img").src = j.origUrl;
      j.el.querySelector(".tags").replaceChildren();
      paint(j);
    }
  }
  stat(); refreshStart();
  await $("start").onclick();
};
$("clear").onclick = () => {
  if (running) return;
  for (const j of jobs){
    if (j.timer){ clearInterval(j.timer); j.timer = null; }
    URL.revokeObjectURL(j.origUrl);
    if (j.censUrl && j.censUrl.startsWith("blob:")) URL.revokeObjectURL(j.censUrl);
  }
  jobs.length = 0;
  $("grid").replaceChildren();
  $("empty").style.display = "";
  stat(); refreshStart();
};

/* ---------- 一键打包下载全部结果 ---------- */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++){
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf){
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
// 生成 STORE(无压缩) 格式 zip；name 需为 ASCII（中文名用 UTF-8 标志）
function buildZip(entries){ // entries: [{name, data: Uint8Array}]
  const enc = new TextEncoder();
  const chunks = [], central = [];
  let offset = 0;
  for (const e of entries){
    const nameU8 = enc.encode(e.name);
    const crc = crc32(e.data);
    const n = e.data.length;
    // local file header (30 bytes)
    const head = new DataView(new ArrayBuffer(30));
    head.setUint32(0, 0x04034b50, true);   // PK\x03\x04
    head.setUint16(4, 20, true);           // version needed
    head.setUint16(6, 0x0800, true);       // general purpose (UTF-8)
    head.setUint16(8, 0, true);            // method: STORE
    head.setUint16(10, 0, true);           // mod time
    head.setUint16(12, 0, true);           // mod date
    head.setUint32(14, crc, true);         // CRC-32
    head.setUint32(18, n, true);           // compressed size
    head.setUint32(22, n, true);           // uncompressed size
    head.setUint16(26, nameU8.length, true); // filename length
    head.setUint16(28, 0, true);           // extra length
    chunks.push(new Uint8Array(head.buffer), nameU8, e.data);
    // central directory header (46 bytes)
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true);    // PK\x01\x02
    cen.setUint16(4, 20, true);            // version made by
    cen.setUint16(6, 20, true);            // version needed
    cen.setUint16(8, 0x0800, true);        // flags
    cen.setUint16(10, 0, true);            // method
    cen.setUint16(12, 0, true);            // time
    cen.setUint16(14, 0, true);            // date
    cen.setUint32(16, crc, true);          // CRC-32
    cen.setUint32(20, n, true);            // compressed
    cen.setUint32(24, n, true);            // uncompressed
    cen.setUint16(28, nameU8.length, true);// filename length
    cen.setUint16(30, 0, true);            // extra length
    cen.setUint16(32, 0, true);            // comment length
    cen.setUint16(34, 0, true);            // disk number
    cen.setUint16(36, 0, true);            // internal attrs
    cen.setUint32(38, 0, true);            // external attrs
    cen.setUint32(42, offset, true);       // local header offset
    central.push({ cen, nameU8 });
    offset += 30 + nameU8.length + n;
  }
  const cdStart = offset;
  let cdSize = 0;
  for (const c of central){ chunks.push(new Uint8Array(c.cen.buffer), c.nameU8); cdSize += 46 + c.nameU8.length; }
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);      // PK\x05\x06
  end.setUint16(8, central.length, true);  // total entries (disk)
  end.setUint16(10, central.length, true); // total entries
  end.setUint32(12, cdSize, true);         // central dir size
  end.setUint32(16, cdStart, true);        // central dir offset
  end.setUint16(20, 0, true);              // comment length
  chunks.push(new Uint8Array(end.buffer));
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks){ out.set(c, p); p += c.length; }
  return out;
}
$("downloadAll").onclick = async () => {
  const done = jobs.filter(j => j.st === "done");
  if (!done.length){ $("stat").textContent = "还没有已完成的结果，先打码再打包"; return; }
  $("downloadAll").disabled = true;
  $("stat").textContent = "正在打包 " + done.length + " 个文件…";
  try {
    const entries = [];
    for (const j of done){
      // 图片结果：base64 dataURL；GIF/视频：blob URL
      const ext = (j.kind === "gif" ? "gif" : j.kind === "video" ? (j.res && j.res.ext) || "mp4" : (j.res && j.res.fmt) || "jpg");
      const baseName = j.file.name.replace(/\\.[^.]+$/, "") + safeSuffix() + "." + ext;
      let buf;
      if (j.censUrl.startsWith("data:")){
        const b64 = j.censUrl.split(",")[1];
        const bin = atob(b64);
        buf = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      } else {
        const blob = await (await fetch(j.censUrl)).blob();
        buf = new Uint8Array(await blob.arrayBuffer());
      }
      entries.push({ name: baseName, data: buf });
    }
    const zip = buildZip(entries);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([zip], { type: "application/zip" }));
    a.download = "打码结果_" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".zip";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    $("stat").textContent = "已打包下载 " + done.length + " 个文件";
  } catch (e){
    $("stat").textContent = "打包失败: " + (e.message || e);
  }
  $("downloadAll").disabled = false;
};

/* ---------- 手动编辑器 ---------- */
const ed = { item: null, baseImg: null, w: 0, h: 0, undo: [], tmp: null };
const edFiles = [];
const pcanvas = $("edPaint"), pctx = pcanvas.getContext("2d");

function addEdFile(f){
  const item = { file: f, url: URL.createObjectURL(f), name: f.name,
                 sizeText: fmtSize(f.size) };
  edFiles.push(item);
  const th = document.createElement("img");
  th.src = item.url; th.title = item.name; th.dataset.name = item.name;
  th.onclick = () => loadEditor(item);
  $("edStrip").appendChild(th);
  return item;
}
function loadEditorFromJob(job){ loadEditor(edFiles.find(x => x.file === job.file) || addEdFile(job.file)); }
function loadEditor(item){
  const img = new Image();
  img.onload = () => {
    ed.item = item; ed.baseImg = img;
    ed.w = img.naturalWidth; ed.h = img.naturalHeight;
    $("edBase").src = item.url;
    pcanvas.width = ed.w; pcanvas.height = ed.h;
    $("edName").textContent = item.name;
    $("editor").style.display = "";
    ed.undo = []; ed.tmp = null;
    for (const th of $("edStrip").children)
      th.classList.toggle("cur", th.dataset.name === item.name);
  };
  img.src = item.url;
}

let drawing = false, lastPt = null;
function ptOf(e){
  const r = pcanvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * pcanvas.width / r.width,
           y: (e.clientY - r.top) * pcanvas.height / r.height };
}
function brushState(){
  return {
    type: $("brush").querySelector(".on").dataset.brush,
    size: +$("brushSize").value,
    color: $("brushColor").value,
    strength: +$("strength").value,
  };
}
function pushUndo(){
  const c = document.createElement("canvas");
  c.width = pcanvas.width; c.height = pcanvas.height;
  c.getContext("2d").drawImage(pcanvas, 0, 0);
  ed.undo.push(c);
  if (ed.undo.length > 10) ed.undo.shift();
}
function dab(p, st){
  const r = st.size / 2;
  pctx.save();
  if (st.type === "color"){
    pctx.fillStyle = st.color;
    pctx.beginPath(); pctx.arc(p.x, p.y, r, 0, 7); pctx.fill();
  } else if (st.type === "eraser"){
    pctx.globalCompositeOperation = "destination-out";
    pctx.beginPath(); pctx.arc(p.x, p.y, r, 0, 7); pctx.fill();
  } else if (st.type === "img"){
    const im = brushStampImg;
    if (im && im.naturalWidth){
      const h = st.size, w = st.size * (im.naturalWidth / im.naturalHeight);
      pctx.drawImage(im, p.x - w / 2, p.y - h / 2, w, h);
    }
  } else {
    pctx.beginPath(); pctx.arc(p.x, p.y, r, 0, 7); pctx.clip();
    if (st.type === "mosaic"){
      const b = Math.max(2, Math.round(st.strength / 8));
      const sx = Math.max(0, Math.min(ed.w - 1, p.x - r));
      const sy = Math.max(0, Math.min(ed.h - 1, p.y - r));
      const sw = Math.max(1, Math.min(ed.w - sx, st.size));
      const sh = Math.max(1, Math.min(ed.h - sy, st.size));
      ed.tmp = ed.tmp || document.createElement("canvas");
      ed.tmp.width = Math.max(1, Math.round(sw / b));
      ed.tmp.height = Math.max(1, Math.round(sh / b));
      const tc = ed.tmp.getContext("2d");
      tc.clearRect(0, 0, ed.tmp.width, ed.tmp.height);
      tc.drawImage(ed.baseImg, sx, sy, sw, sh, 0, 0, ed.tmp.width, ed.tmp.height);
      pctx.imageSmoothingEnabled = false;
      pctx.drawImage(ed.tmp, 0, 0, ed.tmp.width, ed.tmp.height, sx, sy, sw, sh);
      pctx.imageSmoothingEnabled = true;
    } else { // blur
      pctx.filter = "blur(" + Math.max(2, Math.round(st.strength / 6)) + "px)";
      pctx.drawImage(ed.baseImg, 0, 0);
      pctx.filter = "none";
    }
  }
  pctx.restore();
}
pcanvas.addEventListener("pointerdown", e => {
  if (!ed.baseImg) return;
  try { e.target.setPointerCapture(e.pointerId); } catch (err) {}
  pushUndo();
  drawing = true;
  lastPt = ptOf(e);
  dab(lastPt, brushState());
});
pcanvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  const p = ptOf(e), st = brushState();
  const step = Math.max(2, st.size / 4);
  const dx = p.x - lastPt.x, dy = p.y - lastPt.y;
  const dist = Math.hypot(dx, dy);
  if (dist < step) return;
  const n = Math.ceil(dist / step);
  for (let k = 1; k <= n; k++)
    dab({ x: lastPt.x + dx * k / n, y: lastPt.y + dy * k / n }, st);
  lastPt = p;
});
for (const ev of ["pointerup", "pointercancel"])
  pcanvas.addEventListener(ev, () => { drawing = false; });

$("edUndo").onclick = () => {
  const c = ed.undo.pop();
  if (!c) return;
  pctx.clearRect(0, 0, pcanvas.width, pcanvas.height);
  pctx.drawImage(c, 0, 0);
};
$("edClear").onclick = () => {
  pushUndo();
  pctx.clearRect(0, 0, pcanvas.width, pcanvas.height);
};
$("edReset").onclick = () => { if (ed.item) loadEditor(ed.item); };
$("edAi").onclick = async () => {
  if (!ed.item){ $("stat").textContent = "先拖入或点击一张图片"; return; }
  $("edAi").disabled = true;
  try {
    const s = readSettings();
    let assetId = null;
    if (s.mode === "img") assetId = await ensureStampAsset();
    const j = await infer(ed.item.file, s, assetId);
    const im = new Image();
    im.onload = () => { ed.baseImg = im; $("edBase").src = j.censored; };
    im.src = j.censored;
    $("stat").textContent = "AI 预打码完成：检测 " + j.detections.length +
      " 处，可用画笔继续补差";
  } catch (e){
    $("stat").textContent = "AI 预打码失败: " + (e.message || e);
  }
  $("edAi").disabled = false;
};
$("edDl").onclick = () => {
  if (!ed.item){ $("stat").textContent = "先拖入或点击一张图片"; return; }
  const out = document.createElement("canvas");
  out.width = ed.w; out.height = ed.h;
  const oc = out.getContext("2d");
  oc.drawImage($("edBase"), 0, 0, out.width, out.height);
  oc.drawImage(pcanvas, 0, 0);
  out.toBlob(b => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = ed.item.file.name.replace(/\\.[^.]+$/, "") + "_censored.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
};

/* 画笔遮挡图片（本地即可，不上传） */
let brushStampImg = new Image();
$("brushStampPick").onclick = () => $("brushStampFile").click();
$("brushStampFile").onchange = () => {
  const f = $("brushStampFile").files[0];
  if (!f || !f.type.startsWith("image/")) return;
  const url = URL.createObjectURL(f);
  brushStampImg.onload = () => { $("brushStampPrev").src = url;
    $("brushStampPrev").style.display = "block"; };
  brushStampImg.src = url;
};

/* 恢复记住的打码设置 */
if (SET.remember && SET.ai){
  const a = SET.ai;
  if (["mosaic", "blur", "solid", "img"].includes(a.mode)) censorMode = a.mode;
  if (a.strength != null) $("strength").value = a.strength;
  if (a.margin != null) $("margin").value = a.margin;
  if (a.conf != null) $("conf").value = a.conf;
  if (a.res && ALLOWED_RES.includes(a.res)) $("res").value = a.res;
  if (Array.isArray(a.classes) && a.classes.length) renderClasses(a.classes);
  if (typeof a.world === "string") $("worldClasses").value = a.world;
  shown("strength"); shown("margin", "%"); shown("conf");
}
refreshModeUI();

/* ---------- 拖拽与选择 ---------- */
const drop = $("drop"), fileInput = $("file");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });

/* 点击卡片：手动模式图片载入编辑器，其余打开全图预览灯箱 */
$("grid").addEventListener("click", e => {
  if (e.target.closest("a") || e.target.closest("button")) return;
  const card = e.target.closest(".card");
  if (!card) return;
  const job = jobs.find(j => j.el === card);
  if (!job) return;
  if (runMode === "manual" && !job.media && job.st !== "done"){
    loadEditorFromJob(job);
    return;
  }
  openLightbox(job);
});

/* ---------- 灯箱：点击卡片全图预览 ---------- */
const lb = $("lightbox"), lbImg = $("lbMedia"), lbVideo = $("lbVideo"), lbCap = $("lbCap");
function openLightbox(job){
  const media = job.censUrl || job.origUrl;   // 有结果优先展示结果
  if (job.media && job.kind === "video"){
    lbImg.hidden = true; lbVideo.hidden = false;
    lbVideo.src = media; lbVideo.play().catch(() => {});
  } else {
    lbVideo.hidden = true; lbVideo.pause(); lbVideo.removeAttribute("src");
    lbImg.hidden = false; lbImg.src = media;
  }
  lbCap.textContent = job.file.name + (job.censUrl ? "（打码结果）" : "");
  lb.classList.add("show");
  document.body.style.overflow = "hidden";
}
function closeLightbox(){
  lb.classList.remove("show");
  lbVideo.pause(); lbVideo.removeAttribute("src");
  lbImg.removeAttribute("src");
  document.body.style.overflow = "";
}
$("lbClose").onclick = closeLightbox;
lb.addEventListener("click", e => { if (e.target === lb) closeLightbox(); });
addEventListener("keydown", e => { if (e.key === "Escape" && lb.classList.contains("show")) closeLightbox(); });

let dragDepth = 0;
addEventListener("dragover", e => e.preventDefault());
addEventListener("drop", e => e.preventDefault());
drop.addEventListener("dragenter", e => { e.preventDefault(); if (++dragDepth) drop.classList.add("over"); });
drop.addEventListener("dragleave", () => { if (--dragDepth <= 0){ dragDepth = 0; drop.classList.remove("over"); } });
drop.addEventListener("drop", e => {
  e.preventDefault(); dragDepth = 0; drop.classList.remove("over");
  if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});
</script>
</body>
</html>"""
INDEX_HTML = (INDEX_HTML
              .replace("__ALL_CLASSES__", json.dumps(LABELS))
              .replace("__DEFAULT_CLASSES__", json.dumps(DEFAULT_CLASSES))
              .replace("__ZH__", json.dumps({
                  "FEMALE_GENITALIA_COVERED": "女性生殖器·遮挡",
                  "FACE_FEMALE": "女性面部",
                  "BUTTOCKS_EXPOSED": "臀部露出",
                  "FEMALE_BREAST_EXPOSED": "胸部露出",
                  "FEMALE_GENITALIA_EXPOSED": "女性生殖器露出",
                  "MALE_BREAST_EXPOSED": "男性胸部露出",
                  "ANUS_EXPOSED": "肛门露出",
                  "FEET_EXPOSED": "双脚露出",
                  "BELLY_COVERED": "腹部·遮挡",
                  "FEET_COVERED": "双脚·遮挡",
                  "ARMPITS_COVERED": "腋下·遮挡",
                  "ARMPITS_EXPOSED": "腋下露出",
                  "FACE_MALE": "男性面部",
                  "BELLY_EXPOSED": "腹部露出",
                  "MALE_GENITALIA_EXPOSED": "男性生殖器露出",
                  "ANUS_COVERED": "肛门·遮挡",
                  "FEMALE_BREAST_COVERED": "胸部·遮挡",
                  "BUTTOCKS_COVERED": "臀部·遮挡",
              })))
INDEX_HTML = INDEX_HTML.replace("__HASFFMPEG__", "true" if HAS_FFMPEG else "false")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if path in ("/", "/infer"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/video/status":
            jid = (q.get("id", [""])[0] or "")
            with MEDIA_LOCK:
                job = MEDIA.get(jid)
                snap = _media_snapshot(job) if job else None
            if snap is None:
                self._json(404, {"error": "任务不存在"})
            else:
                snap["id"] = jid
                self._json(200, snap)
            return
        if path == "/video/result":
            jid = (q.get("id", [""])[0] or "")
            with MEDIA_LOCK:
                job = MEDIA.get(jid)
                p = job.get("path") if job and job.get("st") == "done" else None
                ext = job.get("ext") if job else None
            if not p or not os.path.exists(p):
                self._json(404, {"error": "结果不存在或已过期"})
                return
            mime = MEDIA_MIME.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(os.path.getsize(p)))
            self.send_header("Content-Disposition",
                             'attachment; filename="censored.%s"' % ext)
            self.end_headers()
            with open(p, "rb") as fh:
                self.wfile.write(fh.read())
            return
        if path == "/lan":
            # 当前局域网访问状态
            self._json(200, {"lan": CURRENT_LAN, "port": CURRENT_PORT,
                             "ips": _lan_ips()})
            return
        self._send(404, b"not found", "text/plain")

    def _cfg_from_query(self, q):
        cfg = build_cfg(
            mode=q.get("mode", ["mosaic"])[0],
            strength=q.get("strength", ["35"])[0],
            margin=q.get("margin", ["15"])[0],
            color=q.get("color", ["#000000"])[0],
            classes=(q.get("classes", [None])[0] or "").split(",")
                    if "classes" in q else None,
            conf=q.get("conf", ["0.25"])[0],
            asset=q.get("asset", [None])[0],
            world_classes=q.get("world", [""])[0],
        )
        try:
            cfg["res"] = int(q.get("res", ["640"])[0])
        except ValueError:
            cfg["res"] = 640
        if cfg["res"] not in ALLOWED_RESOLUTIONS:
            cfg["res"] = 640
        # 输出格式（齿轮设置）：图片 jpg/png + 质量；视频 webm/mp4
        cfg["fmt"] = q.get("fmt", ["jpg"])[0]
        if cfg["fmt"] not in ("jpg", "png"):
            cfg["fmt"] = "jpg"
        try:
            cfg["quality"] = max(40, min(100, int(q.get("quality", ["92"])[0])))
        except ValueError:
            cfg["quality"] = 92
        cfg["vfmt"] = q.get("vfmt", ["webm"])[0]
        if cfg["vfmt"] not in ("webm", "mp4"):
            cfg["vfmt"] = "webm"
        return cfg

    def do_POST(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)

        if parsed.path == "/lan":
            # 切换局域网访问：on=允许, off=仅本机；os.execv 重启服务生效
            mode = q.get("mode", [""])[0]
            target = "lan" if mode == "on" else "local"
            cur = "lan" if CURRENT_LAN else "local"
            if target == cur:
                self._json(200, {"ok": True, "restarting": False,
                                 "lan": CURRENT_LAN})
                return
            # 先把"即将重启"响应发给前端，再替换进程
            body = json.dumps({"ok": True, "restarting": True,
                               "lan": target == "lan",
                               "port": CURRENT_PORT},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")
            self.wfile.flush()
            try:
                _restart_with_mode("lan" if target == "lan" else "local")
            except Exception as e:
                # execv 失败：尽力报错
                try:
                    self._json(500, {"error": "重启失败: %s" % e})
                except Exception:
                    pass
            return

        if parsed.path == "/asset":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > ASSET_LIMIT:
                self._json(400, {"error": "遮挡图片为空或超过 8MB 上限"})
                return
            data = self.rfile.read(length)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._json(400, {"error": "无法解析遮挡图片内容"})
                return
            h, w = img.shape[:2]
            if max(h, w) > 1024:  # 控制内存，等比缩小超大图
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            aid = uuid.uuid4().hex
            with MEDIA_LOCK:
                ASSETS[aid] = {"img": img, "ts": time.time()}
                if len(ASSETS) > ASSET_KEEP:
                    for k in sorted(ASSETS, key=lambda k: ASSETS[k]["ts"])[:-ASSET_KEEP]:
                        ASSETS.pop(k, None)
            self._json(200, {"id": aid, "w": img.shape[1], "h": img.shape[0]})
            return

        cfg = self._cfg_from_query(q)

        if parsed.path == "/infer":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._json(400, {"error": "无法解析图片内容"})
                return
            stamp, err = _resolve_stamp(cfg)
            if err:
                self._json(400, {"error": err})
                return
            t0 = time.time()
            try:
                detections, enabled_world = _detect_combined(img, cfg)
            except RuntimeError as e:
                self._json(400, {"error": str(e)})
                return
            if enabled_world:
                cfg["classes"] = list(cfg["classes"]) + enabled_world
            censored_count = censor_regions(img, detections, cfg, stamp=stamp)
            if cfg["fmt"] == "png":
                ok, buf = cv2.imencode(".png", img)
                mime = "image/png"
            else:
                ok, buf = cv2.imencode(".jpg", img,
                                       [cv2.IMWRITE_JPEG_QUALITY, cfg["quality"]])
                mime = "image/jpeg"
            if not ok:
                self._json(500, {"error": "结果编码失败"})
                return
            elapsed = int((time.time() - t0) * 1000)
            self._json(200, {
                "detections": detections,
                "censored_count": censored_count,
                "elapsed_ms": elapsed,
                "fmt": cfg["fmt"],
                "censored": "data:" + mime + ";base64," + base64.b64encode(buf).decode(),
            })
            return

        if parsed.path == "/video":
            kind = q.get("kind", ["video"])[0]
            if kind not in ("gif", "video"):
                self._json(400, {"error": "kind 仅支持 gif/video"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MEDIA_LIMIT:
                self._json(400, {"error": "文件为空或超过 500MB 上限"})
                return
            stamp, err = _resolve_stamp(cfg)
            if err:
                self._json(400, {"error": err})
                return
            data = self.rfile.read(length)
            jid = uuid.uuid4().hex
            with MEDIA_LOCK:
                MEDIA[jid] = {"id": jid, "ts": time.time(), "st": "processing",
                              "pct": 0, "kind": kind}
            threading.Thread(target=_media_worker, args=(jid, data, kind, cfg, stamp),
                             daemon=True).start()
            self._json(200, {"id": jid, "kind": kind})
            return

        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


if __name__ == "__main__":
    import argparse as _argparse
    ap = _argparse.ArgumentParser(description="打码工作台")
    ap.add_argument("port", nargs="?", type=int, default=8080,
                    help="端口（默认 8080）")
    ap.add_argument("--lan", action="store_true",
                    help="允许局域网访问（绑定 0.0.0.0）")
    args = ap.parse_args()
    port = args.port
    bind_host = LAN_HOST if args.lan else LOOPBACK_HOST
    CURRENT_PORT = port       # 模块级变量，供 /lan 端点使用
    CURRENT_LAN = args.lan    # 是否允许局域网访问

    server = ThreadingHTTPServer((bind_host, port), Handler)
    mode = "局域网" if args.lan else "仅本机"
    print(f"NudeNet 自动打码网页版已启动 ({mode}) : http://localhost:{port}")
    if args.lan:
        for ip in _lan_ips():
            print(f"  局域网访问: http://{ip}:{port}")
    print("  Ctrl+C 退出")
    server.serve_forever()
