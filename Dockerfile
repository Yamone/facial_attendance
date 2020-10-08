FROM python:3.6
COPY . /facial_attendance_server
WORKDIR /facial_attendance_server
RUN ln -s /private/var/mysql/mysql.sock /tmp/mysql.sock
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip install -r requirements.txt
EXPOSE 5005
CMD python ./src/app.py
