import requests
from loguru import logger
r = requests.get("http://127.0.0.1:8000/post/recommendations/")
logger.info(r.status_code)
logger.info(r.text)