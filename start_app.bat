@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Interactive PDF Assistant...
echo Please ensure Ollama is running (e.g., 'ollama serve') and you have pulled a model (e.g., 'ollama pull llama3').
echo.
streamlit run app.py
pause
