@echo off

:: Create a virtual environment
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo Environment setup complete.
pause
