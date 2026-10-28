@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   EmoBeat - 基于人脸情绪识别的歌曲推荐系统
echo ============================================
echo.
echo 服务地址: http://localhost:8080
echo 关闭本窗口即可停止服务
echo.

where python >nul 2>nul
if %errorlevel%==0 (
  start "" http://localhost:8080
  python -m http.server 8080
  goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
  start "" http://localhost:8080
  py -3 -m http.server 8080
  goto :eof
)

where node >nul 2>nul
if %errorlevel%==0 (
  start "" http://localhost:8080
  npx --yes http-server -p 8080 -c-1
  goto :eof
)

echo [错误] 未找到 python 或 node，请先安装其中之一。
pause
