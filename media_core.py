# 媒体打码核心：GIF 与视频的逐帧检测/打码/重编码
# 路径安全说明：所有文件路径均为代码内常量，用 pathlib 在 tempfile 创建的
# 目录下拼出；客户端文件名/扩展名不进入任何路径。
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from censor_core import censor_regions

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _find_ffmpeg():
    """定位可用的 ffmpeg：优先 imageio-ffmpeg 自带二进制（pip 装即可，
    无需系统级安装），其次系统 PATH。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _default_detect_every(kind, n):
    """检测帧距：视频约每秒 6 次检测；GIF 全程约 10 次。其余帧沿用最近检测框。"""
    if kind == "gif":
        return max(1, n // 10)
    return max(1, min(15, round(n / 6)))


def process_gif(data, cfg, detector, stamp=None, detect_every=None, progress=None,
                world=None, world_embeds=None):
    """GIF 打码：PIL 逐帧检测/打码后重新编码为 GIF。返回 (bytes, "gif", info)

    world: 可选 yolo_world.WorldDetector（启用自定义类别时传入），
    world_embeds: 对应的文本嵌入（多帧复用，避免每帧重算）。
    """
    if not HAS_PIL:
        raise RuntimeError("处理 GIF 需要 Pillow（pip install pillow）")
    im = Image.open(io.BytesIO(data))
    n = getattr(im, "n_frames", 1)
    every = detect_every or _default_detect_every("gif", n)
    dets = []
    frames, durations = [], []
    for i in range(n):
        im.seek(i)
        durations.append(int(im.info.get("duration", 100)))
        bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        if i % every == 0:
            dets = detector.detect(bgr, conf=cfg["conf"])
            if world is not None and cfg.get("world_classes"):
                dets = dets + world.detect(bgr, cfg["world_classes"],
                                           embeds=world_embeds,
                                           conf=cfg.get("world_conf", 0.15))
        censor_regions(bgr, dets, cfg, stamp=stamp)
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if progress:
            progress(i + 1, n)
    loop = im.info.get("loop", 0)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True,
                   append_images=frames[1:] if n > 1 else [],
                   duration=durations, loop=loop, disposal=2)
    return out.getvalue(), "gif", {"frames": n, "detect_every": every}


def process_video(data, cfg, detector, stamp=None, detect_every=None, progress=None,
                  prefer="webm", world=None, world_embeds=None):
    """视频打码：OpenCV 逐帧处理。

    输入统一写入临时目录内的固定名文件（OpenCV 的 ffmpeg 后端按内容探测容器，
    不依赖扩展名，客户端文件名不进入任何路径）。
    编码优先级：有 ffmpeg（imageio-ffmpeg 自带或系统）-> 一律 H.264 mp4
    （含音频，重编码 aac）；无 ffmpeg 时按 prefer 选择：webm（VP80，浏览器
    可预览，无音频，失败退 mp4v）或直接 mp4（mp4v，体积小但通常无法预览）。
    world/world_embeds: 启用 YOLO-World 自定义类别时传入（同 GIF）。
    返回 (bytes, 扩展名, info)
    """
    ff = _find_ffmpeg()
    has_ff = ff is not None
    work = Path(tempfile.mkdtemp(prefix="censor_vid_"))
    try:
        src = work / "in.bin"
        src.write_bytes(data)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("无法解码该视频（容器或编码格式不受支持）")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-3 else 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap.release()
            raise RuntimeError("视频尺寸读取失败")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        every = detect_every or _default_detect_every("video", fps)

        proc = writer = outpath = None
        out_ext = None
        if has_ff:
            outpath = work / "out.mp4"
            out_ext = "mp4"
            proc = subprocess.Popen(
                [ff, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "pipe:0",
                 "-i", str(src),
                 "-map", "0:v:0", "-map", "1:a:0?",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-c:a", "aac", "-b:a", "128k", "-shortest", str(outpath)],
                stdin=subprocess.PIPE)
        else:
            if prefer == "mp4":
                outpath = work / "out.mp4"
                out_ext = "mp4"
                writer = cv2.VideoWriter(str(outpath),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
            else:
                outpath = work / "out.webm"
                out_ext = "webm"
                writer = cv2.VideoWriter(str(outpath),
                                         cv2.VideoWriter_fourcc(*"VP80"),
                                         fps, (w, h))
                if not writer.isOpened():
                    outpath = work / "out.mp4"
                    out_ext = "mp4"
                    writer = cv2.VideoWriter(str(outpath),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (w, h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("无法初始化视频编码器（OpenCV 编码后端缺失）")

        dets = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                dets = detector.detect(frame, conf=cfg["conf"])
                if world is not None and cfg.get("world_classes"):
                    dets = dets + world.detect(frame, cfg["world_classes"],
                                               embeds=world_embeds,
                                               conf=cfg.get("world_conf", 0.15))
            censor_regions(frame, dets, cfg, stamp=stamp)
            if has_ff:
                proc.stdin.write(frame.tobytes())
            else:
                writer.write(frame)
            i += 1
            if progress:
                progress(i, total or i)
        cap.release()
        if has_ff:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg 编码失败（未安装或参数不受支持）")
        else:
            writer.release()
        info = {"frames": i, "fps": round(fps, 2), "detect_every": every,
                "audio": has_ff}
        return outpath.read_bytes(), out_ext, info
    finally:
        shutil.rmtree(work, ignore_errors=True)
