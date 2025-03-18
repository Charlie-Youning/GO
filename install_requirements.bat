@echo off
echo Installing packages individually...

REM First install wheel and setuptools
pip install --upgrade pip wheel setuptools -i https://mirrors.aliyun.com/pypi/simple/

REM Install numpy and pandas using wheels
pip install numpy==1.26.4 --only-binary :all: -i https://mirrors.aliyun.com/pypi/simple/
pip install pandas==2.0.3 --only-binary :all: -i https://mirrors.aliyun.com/pypi/simple/

REM Install other dependencies
pip install matplotlib==3.7.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install nltk==3.8.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install vaderSentiment==3.3.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install music21==8.3.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install pygame==2.5.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install Flask==2.3.3 -i https://mirrors.aliyun.com/pypi/simple/
pip install opencv-python==4.8.0.76 -i https://mirrors.aliyun.com/pypi/simple/

REM Install Flask dependencies
pip install Werkzeug==2.3.7 -i https://mirrors.aliyun.com/pypi/simple/
pip install click==8.1.7 -i https://mirrors.aliyun.com/pypi/simple/
pip install colorama==0.4.6 -i https://mirrors.aliyun.com/pypi/simple/
pip install itsdangerous==2.1.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install Jinja2==3.1.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install MarkupSafe==2.1.3 -i https://mirrors.aliyun.com/pypi/simple/

REM Install audio processing and machine learning dependencies
pip install librosa==0.10.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install torch==2.2.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install numba==0.59.0 -i https://mirrors.aliyun.com/pypi/simple/

echo Installation complete!
pause 