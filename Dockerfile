FROM python:3.9

# 
WORKDIR /hog

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# 
COPY ./requirements.txt /hog/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /hog/requirements.txt

# 
COPY ./app /hog/app


ENV PYTHONPATH "${PYTHONPATH}:/hog"

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]