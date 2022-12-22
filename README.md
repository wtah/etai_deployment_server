# ETAI Example Deployment Server
Basic webserver application for general deployment of ai models.


## How to start:
Create virtualenv and install dependencies
eg. 
```shell
pip install -r install/requirements.txt && pip install -r install/additional_requirements.txt
```
Then migrate webserver: 
```shell
python manage.py makemigrations && python manage.py migrate
```

Then start webserver 
```shell
python manage.py runserver 127.0.0.1:8000
```
Now you have two options (your webserver must be running): 

1. For easy testing open your browser and go to 127.0.0.1:8000
2. For a real scenario go to /examples and look at the test scripts for usage from another python script

## Where to insert your models:
`./model_deployment/views.py`
Pick the View that is appropriate for you (Text/Image)

## What to configure:
`./etai_deployment_server/settings.py`

Set <b>INFERENCE_MODE</b> to either 'image' or 'text' depending on your use case. <br />
Set <b>DO_SAVE_PREDICTIONS</b> to either True or False <br />
In case of real deployment turn of <b>DEBUG</b>.


## If you are having problems with the database

Delete db.sqlite3 in the main directory <br />
Run again:
```shell
python manage.py makemigrations && python manage.py migrate
```