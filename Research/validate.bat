@echo off
rem Validate the pipeline or a trained material package. Runs from anywhere.
rem Examples:
rem   validate.bat --selftest
rem   validate.bat PBR\data\material\export.npz
"%~dp0.venv\Scripts\python.exe" "%~dp0trainer\validate.py" %*
