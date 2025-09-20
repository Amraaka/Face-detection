
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://amaraabokhbat_db_user:ZmjFB0JMMVzBh9iG@driver.gycnfna.mongodb.net/?retryWrites=true&w=majority&appName=driver"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)