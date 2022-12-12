import os
import logging
import string
import scipy
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from aiogram import Bot, executor, Dispatcher, types
from bot_api import TOKEN
from joblib import load

tfid = load('tfid.pkl')
lrc = load('lrc.pkl')

def clean(text):
    text = text.lower() 
    text = re.sub(r'http\S+', " ", text) 
    text = re.sub(r'#\w+', ' ', text) 
    text = re.sub(r'\d+', ' ', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'<.*?>',' ', text) 
    return text

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    text = f"Hello, {user_name}!"
    logging.info(f"{user_name=} {user_id=} sent message: {message.text}")
    
    await message.reply(text)

@dp.message_handler()
async def send_class(message: types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    text = message.text
    clean_text = clean(text)
    clean_text=[clean_text]
    transformed_text = tfid.transform(clean_text)
    result = lrc.predict(transformed_text)
    await bot.send_message(user_id, result)


if __name__ == '__main__':
    executor.start_polling(dp)


