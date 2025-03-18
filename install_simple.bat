@echo off
echo Installing packages individually...

REM Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools -i https://mirrors.aliyun.com/pypi/simple/

REM Install main dependencies without version constraints
pip install numpy pandas matplotlib -i https://mirrors.aliyun.com/pypi/simple/
pip install nltk vaderSentiment -i https://mirrors.aliyun.com/pypi/simple/
pip install music21 pygame -i https://mirrors.aliyun.com/pypi/simple/
pip install Flask opencv-python -i https://mirrors.aliyun.com/pypi/simple/

echo Installation complete!
pause 