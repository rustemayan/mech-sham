FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Strategy switch for A/B tests: polite = fight only mechs that attack us; aggro = attack anyone in sight
ENV MECH_POLICY=polite
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
