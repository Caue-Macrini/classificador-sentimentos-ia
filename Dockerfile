FROM python:3.12-slim
WORKDIR /app
COPY classificador_sentimentos.py dados_treino.py ./
CMD ["python", "classificador_sentimentos.py"]
