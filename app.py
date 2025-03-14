from dotenv import load_dotenv
import os

load_dotenv()  # Load secrets from .env file

api_key = os.getenv('API_KEY')
db_password = os.getenv('DB_PASSWORD')
print(f"API Key: {api_key}, DB Password: {db_password}")