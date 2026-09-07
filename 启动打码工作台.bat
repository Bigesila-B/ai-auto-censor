@echo off
chcp 65001 >nul
title 打码工作台
cd /d "%~dp0"

rem 端口：默认 8080，可带参数指定，如  启动打码工作台.bat 9000
set PORT=8080
if not "%~1"=="" set PORT=%~1

rem 优先 python，其次 py 启动器
set PY=python
where python >nul 2>nul || set PY=py
where %PY% >nul 2>nul || (echo [错误] 未找到 Python，请安装 Python 3.10+ 并勾选 "Add Python to PATH" & pause & exit /b 1)

echo ==============================================
echo   打码工作台   http://localhost:%PORT%
echo   关闭本窗口或按 Ctrl+C 即可停止服务
echo ==============================================
rem 若端口被本应用的旧实例（webui.py）占用，自动结束它
set OLDPID=
for /f "tokens=5" %%p in ('netstat -ano ^| find ":%PORT% " ^| find "LISTENING"') do set OLDPID=%%p
if defined OLDPID powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=%OLDPID%'; if($p -and $p.CommandLine -match 'webui\.py'){Stop-Process -Id %OLDPID% -Force; exit 0}else{exit 1}"
if defined OLDPID if not errorlevel 1 (
    echo [提示] 已自动关闭占用端口 %PORT% 的旧实例 ^(PID %OLDPID%^)
    timeout /t 1 /nobreak >nul
)
set OLDPID=
echo [1/2] 检查依赖（缺失会自动安装，首次较慢）…
%PY% check_deps.py
if errorlevel 1 (
    echo [错误] 依赖检查未通过，请按上方日志处理后重试
    pause
    exit /b 1
)
echo [2/2] 启动服务…
rem 3 秒后自动打开浏览器（等服务就绪）
start "" /min cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:%PORT%"
%PY% webui.py %PORT%

echo.
echo [服务已退出] 若上方提示端口被占用，说明已有一个实例在运行，直接用浏览器打开 http://localhost:%PORT% 即可，或换端口启动。
pause
