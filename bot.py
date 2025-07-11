# bot.py
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.utils import executor
import os

# === ТВОЙ Telegram API Token ===
API_TOKEN = '8088455823:AAHr-KuyhhiKAxgO9HDUzhoVtP2A9oyGYRw'

# === Запуск бота ===
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# === Анализ изображения ===
def analyze_image(image_path):
    detector = MTCNN()
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(rgb_img)

    for face in detections:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = rgb_img[y:y+h, x:x+w]

        try:
            analysis = DeepFace.analyze(face_img, actions=['age', 'emotion'], enforce_detection=False)
            age = analysis[0]['age']
            emotion = analysis[0]['dominant_emotion']
            label = f"{int(age)} лет, {emotion}"
        except Exception as e:
            label = "Ошибка анализа"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    result_path = "result.jpg"
    cv2.imwrite(result_path, img)
    return result_path

# === Команда /start ===
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("👋 Привет! Отправь мне фото, и я покажу возраст и настроение по лицу.")

# === Обработка фото ===
@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    await message.photo[-1].download(destination_file="input.jpg")
    await message.reply("🔍 Обрабатываю изображение...")

    result_path = analyze_image("input.jpg")
    await message.reply_photo(InputFile(result_path))

# === Запуск бота ===
if __name__ == '__main__':
    print("Бот запущен...")
    executor.start_polling(dp, skip_updates=True)
