# 用法: python auto_censor.py 图片1.jpg 图片2.png [选项]
# 功能: 检测图片中的隐私部位并打码，输出 *_censored.jpg
# 选项:
#   --mode mosaic|blur|solid   打码方式（默认 mosaic 马赛克）
#   --strength 1-100           强度: 马赛克块大小/模糊核（默认 35）
#   --margin 0-60              检测框边缘外扩百分比（默认 15）
#   --color RRGGBB             纯色填充颜色（默认 000000 黑色）
#   --conf 0.04-0.9            检测置信度阈值，越低越敏感（默认 0.25）
#   --res 320|640|960|1280     推理分辨率，越高越准越慢（默认 640）
#   --classes A,B,...          打码类别；"all"=全部；缺省=隐私部位五类
#   --world "gun,face"         YOLO-World 自定义类别（英文），与 --classes 并用
#   --stamp 图片路径           mode=img 时使用的遮挡图片
#   --fmt jpg|png              输出图片格式（默认 jpg）
#   --quality 40-100           jpg 质量（默认 92）
#   --list-classes             列出全部可用类别后退出
import argparse
import os
import sys

import cv2

from censor_core import (Detector, LABELS, DEFAULT_CLASSES,
                         ALLOWED_RESOLUTIONS, build_cfg, censor_regions)

parser = argparse.ArgumentParser(description="NudeNet 自动打码（640m 模型）")
parser.add_argument("images", nargs="*", help="要处理的图片路径")
parser.add_argument("--mode", default="mosaic",
                    help="打码方式: mosaic(马赛克)/blur(高斯模糊)/solid(纯色)")
parser.add_argument("--strength", type=int, default=35,
                    help="强度 1-100: 马赛克块大小/模糊核")
parser.add_argument("--margin", type=int, default=15, help="检测框边缘外扩 %%")
parser.add_argument("--color", default="000000", help="纯色填充颜色 RRGGBB")
parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
parser.add_argument("--res", type=int, default=640, help="推理分辨率")
parser.add_argument("--classes", default=None,
                    help='打码类别，逗号分隔；"all"=全部；缺省=隐私部位')
parser.add_argument("--world", default=None,
                    help='YOLO-World 自定义类别（英文，逗号分隔，如 "gun,face"）')
parser.add_argument("--stamp", default=None,
                    help="mode=img 时的遮挡图片路径")
parser.add_argument("--fmt", default="jpg", choices=["jpg", "png"],
                    help="输出图片格式")
parser.add_argument("--quality", type=int, default=92, help="jpg 质量 40-100")
parser.add_argument("--list-classes", action="store_true", help="列出全部类别")
args = parser.parse_args()

if args.list_classes:
    print("可用类别（* 为默认打码）:")
    for c in LABELS:
        print(f"  {'*' if c in DEFAULT_CLASSES else ' '} {c}")
    sys.exit(0)

if not args.images:
    parser.print_help()
    sys.exit(1)

if args.res not in ALLOWED_RESOLUTIONS:
    print(f"无效分辨率 {args.res}，可选: {ALLOWED_RESOLUTIONS}")
    sys.exit(1)

classes = None
if args.classes is not None:
    classes = ["all"] if args.classes.strip().lower() == "all" \
        else [c.strip() for c in args.classes.split(",") if c.strip()]

cfg = build_cfg(mode=args.mode, strength=args.strength, margin=args.margin,
                color=args.color, classes=classes, conf=args.conf,
                world_classes=args.world)
stamp = None
if args.stamp:
    stamp = cv2.imread(args.stamp)
    if stamp is None:
        print(f"遮挡图片无法读取: {args.stamp}")
        sys.exit(1)
elif args.mode == "img":
    print("mode=img 需要提供 --stamp 遮挡图片")
    sys.exit(1)
detector = Detector(inference_resolution=args.res)
world_detector = world_embeds = None
if cfg["world_classes"]:
    from yolo_world import (get_world_detector, expand_words,
                            WORLD_RESOLUTION, WORLD_PREFIX)
    world_words = expand_words(cfg["world_classes"])
    print(f"YOLO-World 自定义类别: {world_words}（首次使用需下载模型）")
    world_detector = get_world_detector()
    world_detector.resolution = max(WORLD_RESOLUTION, args.res)
    world_embeds = world_detector.get_embeds(world_words)
    cfg["world_classes"] = world_words
    cfg["classes"] = list(cfg["classes"]) + [WORLD_PREFIX + w
                                             for w in world_words]

for path in args.images:
    img = cv2.imread(path)
    if img is None:
        print(f"{path}: 无法读取，跳过")
        continue
    detections = detector.detect(img, conf=cfg["conf"])
    if world_detector is not None:
        detections = detections + world_detector.detect(
            img, cfg["world_classes"], embeds=world_embeds, conf=cfg["conf"])
    print(f"\n{path}: 检测到 {len(detections)} 个目标")
    for d in sorted(detections, key=lambda v: -v["score"]):
        print(f"  {d['class']}  score={d['score']:.2f}  box={d['box']}")
    n = censor_regions(img, detections, cfg, stamp=stamp)
    stem, _ = os.path.splitext(path)
    quality = max(40, min(100, args.quality))
    out = f"{stem}_censored.{args.fmt}"
    if args.fmt == "png":
        cv2.imwrite(out, img)
    else:
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    color_desc = "纯色" if args.mode == "solid" else \
        ("马赛克" if args.mode == "mosaic" else "高斯模糊")
    print(f"  已打码 {n} 处（{color_desc}）-> {out}")
