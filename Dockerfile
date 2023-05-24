# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app
  
COPY . .

RUN python3 -m pip install --upgrade pip  

RUN pip3 install -r requirements.txt

RUN chmod +x /app/run_app.sh

EXPOSE 8501

ENTRYPOINT ["/bin/bash"]

CMD ["./run_app.sh"]