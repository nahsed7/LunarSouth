# LunarSouth CLPS Mission Planner - Hugging Face Docker Space
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl HOME=/home/user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt
COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false"]
