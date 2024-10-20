FROM ubuntu:22.04

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev
RUN pip install --upgrade pip
RUN pip install fastapi[standard] joblib numpy scikit-learn

WORKDIR /app

COPY . .

CMD ["fastapi","dev", "app.py", "--host", "0.0.0.0"]
