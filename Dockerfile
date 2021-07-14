FROM python:3.9.5

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' attendance-api-user

WORKDIR /opt/attendance_api

ENV FASTAPI_APP app/main.py

# Install requirements, including from Gemfury
ADD ./attendance_api /opt/attendance_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/attendance_api/requirements.txt

RUN chmod +x /opt/attendance_api/run.sh
RUN chown -R attendance-api-user:attendance-api-user ./

USER attendance-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]