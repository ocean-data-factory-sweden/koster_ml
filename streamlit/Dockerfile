FROM python:3.7-slim

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 git -y
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libxvidcore-dev libx264-dev

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

RUN ls -l

EXPOSE 8501

CMD ["streamlit", "run", "app_frontend.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.port",  "8501", "--server.maxUploadSize", "400"]
