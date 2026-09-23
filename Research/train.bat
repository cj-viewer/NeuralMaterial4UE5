@echo off
rem Train a neural material and emit a material package. Runs from anywhere.
rem Examples:
rem   train.bat --data trainer\data\cerberus_4096 --latent-res 512 --out PBR\data\material
rem   train.bat --latent-res 256            (synthetic test set, output under trainer\output\)
rem All arguments are passed straight to trainer\train.py.
"%~dp0.venv\Scripts\python.exe" "%~dp0trainer\train.py" %*
