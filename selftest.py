# 自动化自测：censor_core 单元测试 + HTTP 接口测试 + CLI 冒烟测试
# 运行: python selftest.py   （无需先启动 webui，脚本自起服务）
import base64
import http.client
import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from censor_core import Detector, build_cfg, censor_regions, parse_hex_color

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


print("== 1. parse_hex_color / build_cfg 参数校验 ==")
check("白色 #ffffff -> BGR(255,255,255)", parse_hex_color("#ffffff") == (255, 255, 255))
check("红色 ff0000 -> BGR(0,0,255)", parse_hex_color("ff0000") == (0, 0, 255))
check("3位缩写 #fff -> 白", parse_hex_color("#fff") == (255, 255, 255))
check("非法颜色 -> 黑", parse_hex_color("zzzzzz") == (0, 0, 0))
check("非法 mode -> 回退 mosaic", build_cfg(mode="xxx")["mode"] == "mosaic")
check("strength 超界 -> 100", build_cfg(strength=500)["strength"] == 100)
check("strength 非数字 -> 35", build_cfg(strength="abc")["strength"] == 35)
check("margin 负数 -> 0", build_cfg(margin=-5)["margin"] == 0)
check("margin 超界 -> 60", build_cfg(margin=99)["margin"] == 60)
check("conf 下限钳位 0.04", build_cfg(conf=0.01)["conf"] == 0.04)
check("classes 非法项被过滤", build_cfg(classes=["NOPE", "FACE_MALE"])["classes"] == ["FACE_MALE"])
check("classes all -> 18 类", len(build_cfg(classes=["all"])["classes"]) == 18)
check("img 打码模式被接受", build_cfg(mode="img")["mode"] == "img")
check("asset 合法 id 透传", build_cfg(asset="abc123")["asset"] == "abc123")
check("asset 非法值 -> None", build_cfg(asset="../x")["asset"] is None)

print("== 2. censor_regions 渲染单元测试 ==")
base = np.full((400, 400, 3), 200, np.uint8)
cv2.rectangle(base, (60, 60), (160, 140), (30, 160, 90), -1)  # 一个彩色块当 ROI
box = {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [60, 60, 100, 80]}

img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, color="#000000"))
check("纯色黑填充，ROI 全黑", bool(np.all(img[60:140, 60:160] == 0)))
check("ROI 之外未被改动", bool(np.all(img[0:60, :] == 200)))

img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, color="#ff0000"))
check("自定义红色填充", bool(np.all(img[60:140, 60:160] == (0, 0, 255))))

noise_src = base.copy()
noise_src[60:140, 60:160] = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
img = noise_src.copy()
censor_regions(img, [box], build_cfg(mode="mosaic", strength=35, margin=0))
roi = img[60:140, 60:160]
check("马赛克后 ROI 有变化", not np.array_equal(roi, noise_src[60:140, 60:160]))
scale = max(base.shape[:2]) / 640.0  # 与 censor_regions 同口径：块大小随图片尺寸缩放
blk = max(2, round((2 + 0.62 * 35) * scale))
check("马赛克块内颜色一致",
      len(np.unique(roi[2:blk - 2, 2:blk - 2].reshape(-1, 3), axis=0)) <= 1,
      f"block={blk}")

img_lo = noise_src.copy()
censor_regions(img_lo, [box], build_cfg(mode="mosaic", strength=1, margin=0))
img_hi = noise_src.copy()
censor_regions(img_hi, [box], build_cfg(mode="mosaic", strength=50, margin=0))
u_lo = len(np.unique(img_lo[60:140, 60:160].reshape(-1, 3), axis=0))
u_hi = len(np.unique(img_hi[60:140, 60:160].reshape(-1, 3), axis=0))
check(f"强度=1 的马赛克明显更细腻（细节色数 {u_lo} > {u_hi}）", u_lo > u_hi)

img = noise_src.copy()
var_before = cv2.Laplacian(noise_src[60:140, 60:160], cv2.CV_64F).var()
censor_regions(img, [box], build_cfg(mode="blur", strength=50, margin=0))
var_after = cv2.Laplacian(img[60:140, 60:160], cv2.CV_64F).var()
check(f"高斯模糊降低高频 ({var_before:.0f} -> {var_after:.1f})", var_after < var_before * 0.2)

img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, classes=["FACE_MALE"]))
check("类别不在选择集 -> 不打码", bool(np.array_equal(img, base)))
img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, classes=[]))
check("类别列表为空 -> 不打码", bool(np.array_equal(img, base)))
img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, classes=["all"]))
check("classes=all -> 打码", not np.array_equal(img, base))

img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=0, color="#000000"))
area0 = int(np.any(img != base, axis=2).sum())
img = base.copy()
censor_regions(img, [box], build_cfg(mode="solid", margin=20, color="#000000"))
area20 = int(np.any(img != base, axis=2).sum())
check(f"边缘外扩增大覆盖 ({area0} -> {area20})", area20 > area0 * 1.5)

tiny = {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [0, 0, 2, 2]}
img = base.copy()
n = censor_regions(img, [tiny], build_cfg(mode="blur", strength=50, margin=0, color="#000000"))
check("极小区域模糊退化不崩溃", n == 1)
off = {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [390, 390, 50, 50]}
img = base.copy()
n = censor_regions(img, [off], build_cfg(mode="solid", margin=0, color="#000000"))
check("越界框被裁剪且不崩溃", n == 1 and bool(np.all(img[390:, 390:] == 0)))

print("== 2b. 图片遮挡模式 ==")
stamp = np.full((30, 20, 3), (0, 255, 0), np.uint8)  # 纯绿小图
img = base.copy()
n = censor_regions(img, [box], build_cfg(mode="img", margin=0), stamp=stamp)
check("图片遮挡：区域被绿色图盖住", n == 1 and bool(np.all(img[60:140, 60:160] == (0, 255, 0))))
img = base.copy()
n = censor_regions(img, [box], build_cfg(mode="img", margin=0, color="#123456"), stamp=None)
check("图片遮挡缺图时退化为纯色",
      n == 1 and bool(np.all(img[60:140, 60:160] == parse_hex_color("#123456"))))
big_stamp = np.zeros((200, 200, 3), np.uint8)
big_stamp[:, :100] = (255, 0, 0)  # 左半红右半黑
img = base.copy()
censor_regions(img, [box], build_cfg(mode="img", margin=0), stamp=big_stamp)
roi_mean = img[60:140, 60:160].reshape(-1, 3).mean(axis=0)
# 大图左半 B=255 右半 0，等比缩放后全宽入框，居中裁剪 -> 均值约 (127.5, 0, 0)
check("图片遮挡等比放大且居中（蓝黑各半）",
      110 < roi_mean[0] < 145 and roi_mean[1] < 10 and roi_mean[2] < 10, str(roi_mean))

print("== 3. Detector 检测器 ==")
det = Detector()
flat = np.full((240, 320, 3), 128, np.uint8)
dets = det.detect(flat, conf=0.25)
check("纯色图 0 检测", len(dets) == 0, str(dets))
src = cv2.imread("test.jpg")
dets = det.detect(src, conf=0.04)
check("低阈值运行不崩溃", isinstance(dets, list))
ok_bounds = all(0 <= b["box"][0] and 0 <= b["box"][1] and
                b["box"][0] + b["box"][2] <= src.shape[1] and
                b["box"][1] + b["box"][3] <= src.shape[0] for b in dets)
check("检测框均在图界内", ok_bounds)
det.resolution = 1280
dets_hi = det.detect(src, conf=0.04)
det.resolution = 640
check("分辨率切换生效（1280 候选不少于 640）", len(dets_hi) >= len(dets))

print("== 3b. 长图自动切片 ==")
long_img = np.zeros((2000, 400, 3), np.uint8)   # 5:1 超长图
long_img[:, :, 0] = 80
# 放几个"目标"色块（模拟人物区块）
cv2.rectangle(long_img, (120, 200), (280, 500), (200, 200, 200), -1)
cv2.rectangle(long_img, (120, 900), (280, 1200), (200, 200, 200), -1)
cv2.rectangle(long_img, (120, 1500), (280, 1800), (200, 200, 200), -1)
dets_long = det.detect(long_img, conf=0.05)
check("长图切片检测运行不崩溃", isinstance(dets_long, list))
ok_bounds = all(0 <= b["box"][0] and 0 <= b["box"][1] and
                b["box"][0] + b["box"][2] <= long_img.shape[1] and
                b["box"][1] + b["box"][3] <= long_img.shape[0]
                for b in dets_long)
check("长图检测框全部在图界内", ok_bounds, str(dets_long[:3]))
# 普通图仍走原路径（不切片）
normal = np.full((600, 800, 3), 100, np.uint8)
dets_n = det.detect(normal, conf=0.05)
check("常规图不触发切片（0 检测）", len(dets_n) == 0)

print("== 4. HTTP 接口 ==")
import webui  # 会创建 640m 检测器

server = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
# 测试目标是脚本自己启动的本机回环服务，主机名固定为字面量 127.0.0.1
conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=60)

conn.request("GET", "/")
resp = conn.getresponse()
html = resp.read().decode("utf-8")
check("首页 200 且包含设置控件",
      resp.status == 200 and
      all(k in html for k in ('id="mode"', 'id="color"', 'id="strength"',
                              'id="margin"', 'id="conf"', 'id="res"', 'id="classes"',
                              'id="theme"')))

jpg = open("test.jpg", "rb").read()


def post(qs, data=jpg):
    conn.request("POST", "/infer" + qs, body=data)
    r = conn.getresponse()
    body = r.read()
    return r.status, json.loads(body)


code, j = post("")
check("默认参数返回 200", code == 200)
check("响应字段齐全",
      all(k in j for k in ("detections", "censored_count", "elapsed_ms", "censored")))
out = cv2.imdecode(np.frombuffer(base64.b64decode(j["censored"].split(",")[1]), np.uint8), 1)
check("返回图与原图同尺寸", out is not None and out.shape == src.shape)

code, j = post("?mode=solid&color=%23ff0000&strength=99&margin=40&conf=0.05&res=320&classes=all")
check("全参数 solid 红 + all 类别 返回 200", code == 200 and j["censored_count"] >= 0)
code, j = post("?mode=mosaic&conf=0.04&res=1280")
check("低阈值 + 1280 分辨率返回 200", code == 200)
code, j = post("", data=b"not-an-image")
check("非法图片返回 400", code == 400 and "error" in j)
code, j = post("?mode=solid&strength=abc&conf=xyz&res=7777", data=jpg)
check("非法参数被钳位不报错", code == 200)

print("== 4b. 图片遮挡接口 ==")
import io as _io4
from PIL import Image as PILImage4
png_buf = _io4.BytesIO()
PILImage4.new("RGB", (20, 20), (0, 255, 0)).save(png_buf, format="PNG")
conn.request("POST", "/asset", body=png_buf.getvalue())
r_ = conn.getresponse()
asset_resp = json.loads(r_.read())
check("上传遮挡图返回 id", r_.status == 200 and bool(asset_resp.get("id")),
      str(asset_resp))
aid = asset_resp["id"]
code, j = post("?mode=img&asset=" + aid)
check("mode=img + asset 推理正常", code == 200)
code, j = post("?mode=img")
check("mode=img 缺 asset 返回 400", code == 400 and "遮挡" in j.get("error", ""), str(j))
conn.request("POST", "/asset", body=b"junk-not-an-image")
r_ = conn.getresponse()
check("坏遮挡图返回 400", r_.status == 400)

print("== 4c. 输出格式设置 ==")
code, j = post("?fmt=png")
check("fmt=png 返回 PNG", code == 200 and j["censored"].startswith("data:image/png"))
out_png = cv2.imdecode(np.frombuffer(base64.b64decode(
    j["censored"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
check("PNG 可解码", out_png is not None and out_png.shape == src.shape)
code, j = post("?fmt=jpg&quality=60")
check("fmt=jpg quality 生效", code == 200 and j["censored"].startswith("data:image/jpeg"))
code, j = post("?fmt=bogus&quality=999")
check("非法 fmt/quality 钳位", code == 200 and j["censored"].startswith("data:image/jpeg"))

print("== 5. CLI 冒烟测试 ==")
r = subprocess.run([sys.executable, "auto_censor.py", "--list-classes"],
                   capture_output=True, text=True)
check("--list-classes 正常", r.returncode == 0 and "FEMALE_BREAST_EXPOSED" in r.stdout)
r = subprocess.run([sys.executable, "auto_censor.py", "test.jpg",
                    "--mode", "solid", "--color", "00ff00", "--margin", "30"],
                   capture_output=True, text=True)
check("CLI 处理 test.jpg 正常", r.returncode == 0 and "已打码" in r.stdout,
      (r.stdout + r.stderr)[-300:])
check("输出文件已生成", cv2.imread("test_censored.jpg") is not None)
r = subprocess.run([sys.executable, "auto_censor.py", "--mode", "bad"],
                   capture_output=True, text=True)
check("CLI 无图片参数给出帮助", r.returncode != 0 and "usage" in (r.stdout + r.stderr).lower())

print("== 6. GIF / 视频媒体任务 ==")
import io as _io


def post_media(qs, data):
    conn.request("POST", "/video" + qs, body=data)
    r_ = conn.getresponse()
    body = r_.read()
    return r_.status, json.loads(body)


def wait_media(mid, timeout=90):
    import time as _t
    dl = _t.time() + timeout
    while _t.time() < dl:
        conn.request("GET", "/video/status?id=" + mid)
        r_ = conn.getresponse()
        st = json.loads(r_.read())
        if st["st"] in ("done", "fail"):
            return st
        _t.sleep(0.3)
    return {"st": "timeout"}


def get_result(mid):
    conn.request("GET", "/video/result?id=" + mid)
    r_ = conn.getresponse()
    return r_.status, r_.read(), r_.getheader("Content-Type")


# 合成 6 帧 GIF
from PIL import Image as PILImage
gif_buf = _io.BytesIO()
frames = [PILImage.new("RGB", (120, 100), (30 * k, 120, 200))
          for k in range(6)]
frames[0].save(gif_buf, format="GIF", save_all=True,
               append_images=frames[1:], duration=80, loop=0)
gif_bytes = gif_buf.getvalue()

code, j = post_media("?kind=gif&mode=solid&color=ff0000&conf=0.1", gif_bytes)
check("GIF 任务提交返回 id", code == 200 and j.get("kind") == "gif" and j.get("id"))
st = wait_media(j["id"])
check("GIF 任务完成", st["st"] == "done", str(st.get("err")))
code, data, ctype = get_result(j["id"])
check("GIF 结果可取且为 image/gif", code == 200 and ctype == "image/gif")
out_gif = PILImage.open(_io.BytesIO(data))
check("GIF 输出帧数与输入一致",
      getattr(out_gif, "n_frames", 1) == 6, f"n_frames={getattr(out_gif, 'n_frames', 1)}")

# 合成 12 帧 mp4
vp = cv2.VideoWriter("_t.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (160, 120))
for k in range(12):
    fr = np.full((120, 160, 3), (20 * k, 90, 60), np.uint8)
    vp.write(fr)
vp.release()
mp4_bytes = open("_t.mp4", "rb").read()
os.remove("_t.mp4")

code, j = post_media("?kind=video&mode=solid", mp4_bytes)
check("视频任务提交返回 id", code == 200 and j.get("kind") == "video" and j.get("id"))
st = wait_media(j["id"])
check("视频任务完成", st["st"] == "done", str(st.get("err")))
check("视频输出帧数一致", st.get("info", {}).get("frames") == 12,
      str(st.get("info")))
code, data, ctype = get_result(j["id"])
check("视频结果可取", code == 200 and len(data) > 100)
magic_ok = data[:4] == b"\x1aE\xdf\xa3" or data[4:8] == b"ftyp"
check("视频容器为 webm/mp4", magic_ok, data[:12].hex())

# 指定 mp4 输出格式
code, j = post_media("?kind=video&mode=solid&vfmt=mp4", mp4_bytes)
st = wait_media(j["id"])
check("vfmt=mp4 任务完成", st["st"] == "done", str(st.get("err")))
code, data, ctype = get_result(j["id"])
check("vfmt=mp4 输出为 mp4 容器", data[4:8] == b"ftyp", data[:12].hex())
ext_out = "webm" if data[:4] == b"\x1aE\xdf\xa3" else "mp4"
tmp_out = Path("_t_result").with_suffix("." + ext_out)
tmp_out.write_bytes(data)
cap = cv2.VideoCapture(str(tmp_out))
n_read = 0
while True:
    ok, _fr = cap.read()
    if not ok:
        break
    n_read += 1
cap.release()
tmp_out.unlink()
check("输出视频可解码且帧数=12", n_read == 12, f"n_read={n_read}")
code2, j2 = post_media("?kind=bogus", b"x")
check("非法 kind 返回 400", code2 == 400)
conn.request("GET", "/video/status?id=nonexistent")
r_ = conn.getresponse()
check("未知任务 404", r_.status == 404)
code2, j2 = post_media("?kind=video", b"not-a-video")
st2 = wait_media(j2["id"]) if code2 == 200 and j2.get("id") else {"st": "skip"}
check("坏视频数据在任务中报失败", st2["st"] == "fail", str(st2))

server.shutdown()
print(f"\n===== 结果: {PASS} 通过, {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)
