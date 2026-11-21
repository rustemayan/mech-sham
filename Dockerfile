FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV MECH_POLICY=polite
# ---- experiment switches: change ONE line per submission ----
ENV STYLE=corner
ENV OPPORTUNIST=0
ENV DIAG=1
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
