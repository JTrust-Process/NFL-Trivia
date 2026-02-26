from app import app, init_db

# Called once by gunicorn --preload before workers fork
init_db()