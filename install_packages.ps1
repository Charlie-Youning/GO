Write-Host "Installing packages individually..."

# Upgrade pip and install wheel
python -m pip install --upgrade pip wheel setuptools -i https://mirrors.aliyun.com/pypi/simple/

# Install main dependencies
python -m pip install numpy --only-binary :all: -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install pandas --only-binary :all: -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install nltk -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install vaderSentiment -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install music21 -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install pygame -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install Flask -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/

# Install Flask dependencies
python -m pip install Werkzeug -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install click -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install colorama -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install itsdangerous -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install Jinja2 -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install MarkupSafe -i https://mirrors.aliyun.com/pypi/simple/

Write-Host "Installation complete!"
pause 