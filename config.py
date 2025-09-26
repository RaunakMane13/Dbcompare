# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env sitting next to main.py/config.py
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

class Settings:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')

    # Paths
    STATIC_DIR   = os.getenv('STATIC_DIR', '/home/student/my-app/src/static')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/home/student/my-app/src/uploads')

    # MySQL (keep your values)
    MYSQL_HOST = os.getenv('MYSQL_HOST', '172.16.1.125')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'StrongPassword123!')

    # Mongo (keep your value)
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://172.16.1.126:27017/')

    # Neo4j (keep your values)
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '12345')

    # Celery (keep current defaults)
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

settings = Settings()
