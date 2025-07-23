@echo off
echo ========================================
echo Sports Project Setup for Windows
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
) else (
    echo Python is installed.
    python --version
)

echo.
echo Step 2: Installing project dependencies...
python -m pip install --upgrade pip
python -m pip install -e .

echo.
echo Step 3: Installing soccer example dependencies...
cd examples\soccer
python -m pip install -r requirements.txt

echo.
echo Step 4: Setting up data directory and downloading models...
if not exist "data" mkdir data

echo Downloading models and videos...
python -m pip install gdown

echo Downloading football-ball-detection.pt...
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V', 'data/football-ball-detection.pt', quiet=False)"

echo Downloading football-player-detection.pt...
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q', 'data/football-player-detection.pt', quiet=False)"

echo Downloading football-pitch-detection.pt...
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf', 'data/football-pitch-detection.pt', quiet=False)"

echo Downloading sample videos...
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf', 'data/2e57b9_0.mp4', quiet=False)"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the project:
echo 1. Open VS Code in the sports directory
echo 2. Open a terminal in VS Code
echo 3. Navigate to examples/soccer
echo 4. Run: python run_example.py
echo.
pause 