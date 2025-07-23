# Running the Sports Project in VS Code

## Prerequisites

1. **Python 3.8+** - Download from https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **VS Code** - Already installed on your system

## Quick Setup

### Option 1: Automatic Setup (Recommended)
1. Open PowerShell as Administrator
2. Navigate to the project directory:
   ```powershell
   cd "C:\Users\Josha\Documents\sports"
   ```
3. Run the setup script:
   ```powershell
   .\setup_windows.ps1
   ```

### Option 2: Manual Setup
If the automatic setup doesn't work, follow these steps:

1. **Install Python dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e .
   cd examples\soccer
   python -m pip install -r requirements.txt
   ```

2. **Download models and data:**
   ```powershell
   python -m pip install gdown
   mkdir data
   python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V', 'data/football-ball-detection.pt', quiet=False)"
   python -c "import gdown; gdown.download('https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q', 'data/football-player-detection.pt', quiet=False)"
   python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf', 'data/football-pitch-detection.pt', quiet=False)"
   python -c "import gdown; gdown.download('https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf', 'data/2e57b9_0.mp4', quiet=False)"
   ```

## Running the Project

1. **Open VS Code:**
   - Open VS Code
   - File → Open Folder → Select `C:\Users\Josha\Documents\sports`

2. **Open Terminal in VS Code:**
   - Terminal → New Terminal
   - Navigate to the soccer example:
   ```bash
   cd examples/soccer
   ```

3. **Run the project:**
   ```bash
   python run_example.py
   ```

## Available Modes

The project supports several analysis modes:

- **PLAYER_DETECTION** - Detect players, goalkeepers, referees, and ball
- **BALL_DETECTION** - Track the ball specifically
- **PITCH_DETECTION** - Detect soccer field boundaries
- **PLAYER_TRACKING** - Track players across frames
- **TEAM_CLASSIFICATION** - Classify players into teams
- **RADAR** - Create radar-like visualization
- **PASS_TRACKING** - Track passes and ball proximity (NEW!)

## Troubleshooting

### Python not found
- Make sure Python is installed and added to PATH
- Try running: `python --version`

### Missing dependencies
- Run: `python -m pip install -r requirements.txt`

### Missing data files
- Run the setup script again or manually download the files

### VS Code not recognizing Python
- Install the Python extension in VS Code
- Select the correct Python interpreter: Ctrl+Shift+P → "Python: Select Interpreter"

## Project Structure

```
sports/
├── examples/soccer/          # Soccer analysis examples
│   ├── data/                # Models and videos (created by setup)
│   ├── run_example.py       # Main runner script
│   └── requirements.txt     # Python dependencies
├── sports/                  # Main package
└── setup.py                # Project configuration
```

## Output

The analysis will create output videos in the `examples/soccer/data/` directory with names like:
- `2e57b9_0-player-detection.mp4`
- `2e57b9_0-ball-detection.mp4`
- `2e57b9_0-pass-tracking.mp4` 