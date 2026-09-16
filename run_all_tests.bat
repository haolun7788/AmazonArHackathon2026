@echo off

python scripts/run_game.py test_cases/level1/test_case_1.json
if errorlevel 1 exit /b 1

python scripts/run_game.py test_cases/level1/test_case_2.json
if errorlevel 1 exit /b 1

python scripts/run_game.py test_cases/level2/test_case_3.json
if errorlevel 1 exit /b 1

python scripts/run_game.py test_cases/level2/test_case_4.json
if errorlevel 1 exit /b 1

python scripts/run_game.py test_cases/level3/test_case_5.json
if errorlevel 1 exit /b 1

python scripts/run_game.py test_cases/level3/test_case_6.json
if errorlevel 1 exit /b 1
