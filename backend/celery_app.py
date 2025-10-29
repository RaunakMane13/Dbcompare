from config import settings
from celery import Celery

celery = Celery(
    'main',
    backend=settings.CELERY_RESULT_BACKEND,
    broker=settings.CELERY_BROKER_URL
)


celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'process_file_task': {'queue': 'default'},
    }
)

celery.autodiscover_tasks(['main'])
# import main  # Ensures process_file_task in main.py is loaded by the worker

