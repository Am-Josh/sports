@echo off
echo ========================================
echo Sports Project Runner
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from the Microsoft Store or https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python is installed!
python --version

echo.
echo Testing Python installation...
python test_python.py

echo.
echo Setting up the project...
python -m pip install --upgrade pip
python -m pip install -e .

echo.
echo Installing soccer example dependencies...
cd examples\soccer
python -m pip install -r requirements.txt

echo.
echo Setting up data directory...
if not exist "data" mkdir data

echo.
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
echo Setup Complete! Running the project...
echo ========================================
echo.

python run_example.py

echo.
echo ========================================
echo Project execution complete!
echo ========================================
echo.
echo Check the data/ directory for output videos.
pause 