#base Image
FROM python:3.11.7

#working directory
WORKDIR /app

#copy 
COPY . /app

#run
RUN pip install -r requirements.txt

#Port 
EXPOSE 5000

#Command to execute or start
CMD [ "python","./app.py" ]
