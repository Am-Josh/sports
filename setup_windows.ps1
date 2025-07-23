Write-Host "========================================" -ForegroundColor Green
Write-Host "Sports Project Setup for Windows" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python is installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Installing project dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -e .

Write-Host ""
Write-Host "Step 3: Installing soccer example dependencies..." -ForegroundColor Yellow
Set-Location "examples\soccer"
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Step 4: Setting up data directory and downloading models..." -ForegroundColor Yellow
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

Write-Host "Downloading models and videos..." -ForegroundColor Yellow
python -m pip install gdown

Write-Host "Downloading football-ball-detection.pt..." -ForegroundColor Yellow
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V', 'data/football-ball-detection.pt', quiet=False)"

Write-Host "Downloading football-player-detection.pt..." -ForegroundColor Yellow
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q', 'data/football-player-detection.pt', quiet=False)"

Write-Host "Downloading football-pitch-detection.pt..." -ForegroundColor Yellow
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf', 'data/football-pitch-detection.pt', quiet=False)"

Write-Host "Downloading sample videos..." -ForegroundColor Yellow
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf', 'data/2e57b9_0.mp4', quiet=False)"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the project:" -ForegroundColor Yellow
Write-Host "1. Open VS Code in the sports directory" -ForegroundColor White
Write-Host "2. Open a terminal in VS Code" -ForegroundColor White
Write-Host "3. Navigate to examples/soccer" -ForegroundColor White
Write-Host "4. Run: python run_example.py" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue" 