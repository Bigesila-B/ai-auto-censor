# 依赖检测与自动安装：python check_deps.py
# 检测 Python 版本、必需 pip 包、模型文件与可选组件（ffmpeg），
# 缺失的 pip 包会自动安装（默认源失败自动切换清华镜像）。
# 注：pkg 只来自本文件内写死的 REQUIRED 清单，不含任何用户输入；
#     安装通过 runpy 在进程内调用 pip，不执行任何 shell 命令。
import importlib
import runpy
import shutil
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("onnxruntime", "onnxruntime"),
    ("PIL", "Pillow"),  # GIF 打码需要
    ("imageio_ffmpeg", "imageio-ffmpeg"),  # 自带 ffmpeg，视频输出带音频
]
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def pip_install(pkg):
    """进程内调用 pip 安装（等价于 python -m pip install）。默认源失败换镜像。"""
    attempts = [
        (["install", "--quiet", pkg], "默认源"),
        (["install", "--quiet", "-i", MIRROR, pkg], "清华镜像"),
    ]
    for args, tag in attempts:
        print(f"  正在安装 {pkg}（{tag}）…")
        old_argv = sys.argv
        sys.argv = ["pip"] + args
        try:
            runpy.run_module("pip", run_name="__main__", alter_sys=True)
            return True   # 正常结束视为成功
        except SystemExit as e:
            if e.code in (0, None):
                return True
        finally:
            sys.argv = old_argv
    return False


def main():
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        print("[错误] 需要 Python 3.9 及以上（推荐 3.10+）")
        return 1

    print("\n== 检查 pip 包 ==")
    missing = []
    for mod, pkg in REQUIRED:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {pkg}")
        except Exception as e:
            print(f"  [缺失] {pkg}（{e.__class__.__name__}）")
            missing.append(pkg)

    if missing:
        print("\n发现缺失依赖：" + "、".join(missing) + "，开始自动安装…")
        for pkg in missing:
            if not pip_install(pkg):
                print(f"[错误] {pkg} 自动安装失败，请手动执行：")
                print(f"  {sys.executable} -m pip install {pkg}")
                print(f"  或使用国内镜像：{sys.executable} -m pip install {pkg} -i {MIRROR}")
                print("  若提示权限不足，可在命令末尾加 --user")
        print("\n== 复检 ==")
        still_missing = []
        for mod, pkg in REQUIRED:
            try:
                importlib.import_module(mod)
                print(f"  [OK]   {pkg}")
            except Exception:
                print(f"  [仍缺失] {pkg}")
                still_missing.append(pkg)
        if still_missing:
            return 1
    else:
        print("  全部齐备")

    # 模型文件（缺 640m 自动下载，失败回退自带的 320n）
    print("\n== 检查模型文件 ==")
    try:
        from censor_core import ensure_model
        model = ensure_model()
        print(f"  [OK]   使用模型 {model}")
    except Exception as e:
        print(f"  [警告] 模型检查未完成：{e}")
        print("         启动服务后仍会自动重试下载")

    # 可选组件
    print("\n== 可选组件 ==")
    try:
        from media_core import _find_ffmpeg
        if _find_ffmpeg():
            print("  [OK]   ffmpeg（视频输出将包含音频）")
        else:
            print("  [提示] 未找到 ffmpeg：视频输出为 WebM 静音格式")
            print("         安装 imageio-ffmpeg 后自动带音频（pip install imageio-ffmpeg）")
    except Exception:
        print("  [提示] ffmpeg 状态未知（视频可能静音）")

    print("\n依赖检查完成 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
