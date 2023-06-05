from tkinter import filedialog
# pip install pyyaml
# !pip install SpeechRecognition
# !pip install vosk
# !pip install pydub
# pip install numpy
# pip install google-cloud-speech
import tkinter as tk

import requests
import io
from pydub import AudioSegment
import vosk
from vosk import Model, KaldiRecognizer
import wave
import json
import re
import os
import string
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import torch
import threading
from tkinter import ttk
from google.cloud import speech
import subprocess


def get_audio_file(audio_link):
    # Отправляем GET-запрос по ссылке и получаем содержимое файла в виде байтов
    response = requests.get(audio_link)
    audio_content = response.content

    # Преобразуем байты в объект BytesIO, который можно передать в библиотеку pydub
    audio_file = io.BytesIO(audio_content)

    # Конвертируем файл в формат wav и возвращаем его из функции
    audio = AudioSegment.from_file(audio_file)
    return audio.export("dialog.wav", format="wav")


def recognize_speech(wavfile):
    # Проверяем, есть ли папка с именем "model" в текущей директории
    if not os.path.exists("vosk-model-ru-0.42"):
        # # Если нет, то загружаем модель с помощью команды wget
        # os.system("wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip")
        # # Распаковываем архив с помощью команды unzip
        # os.system("unzip vosk-model-ru-0.42.zip")
        # # Переименовываем папку с моделью в "model"
        # os.system("mv vosk-model-ru-0.42 model")
        # # Удаляем архив с моделью
        # os.system("rm vosk-model-ru-0.42.zip")
        subprocess.run(["curl", "-L", "-o", "vosk-model-ru-0.42.zip",
                        "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"])
        # Распаковываем архив с помощью команды tar
        subprocess.run(["tar", "-xf", "vosk-model-ru-0.42.zip"], shell=True)
        # Удаляем архив с моделью
        subprocess.run(["del", "vosk-model-ru-0.42.zip"], shell=True)

    # Загружаем модель для русского языка
    model = vosk.Model("vosk-model-ru-0.42")

    # Открываем аудиофайл в формате wav
    wf = wave.open(wavfile, "rb")

    # Получаем частоту дискретизации и количество каналов аудиофайла
    rcgn_fr = wf.getframerate() * wf.getnchannels()

    # Создаем распознаватель речи на основе модели KaldiRecognizer
    rec = KaldiRecognizer(model, rcgn_fr)

    # Инициализируем пустую строку для хранения результата распознавания
    result = ''

    # Инициализируем переменную для отслеживания переносов строк
    last_n = False

    # Задаем размер блока для чтения аудиофайла
    read_block_size = 4000

    # Читаем аудиофайл по блокам в цикле
    while True:
        # Читаем блок данных из аудиофайла
        data = wf.readframes(read_block_size)

        # Если блок данных пустой, то выходим из цикла
        if len(data) == 0:
            break

        # Если распознаватель речи принял блок данных
        if rec.AcceptWaveform(data):
            # Получаем результат распознавания в формате JSON
            res = json.loads(rec.Result())

            # Если результат содержит непустой текст
            if res['text'] != '':
                # Добавляем текст к строке результата с пробелом
                result += f" {res['text']}"

                # Устанавливаем переменную переноса строки в False
                last_n = False

            # Иначе, если переменная переноса строки равна False
            elif not last_n:
                # Добавляем перенос строки к строке результата
                result += '\n'

                # Устанавливаем переменную переноса строки в True
                last_n = True

    # Получаем окончательный результат распознавания в формате JSON
    res = json.loads(rec.FinalResult())

    # Добавляем текст к строке результата с пробелом
    result += f" {res['text']}"

    # Возвращаем строку результата из функции
    return result


def recognize_speech_google(wavfile):
    # Создаем объект распознавателя
    recognizer = sr.Recognizer()
    # Открываем аудиофайл по ссылке
    with sr.AudioFile("dialog.wav") as source:
        # Считываем аудиоданные из файла
        audio_data = recognizer.record(source)
        # Распознаем текст с помощью Google Speech Recognition API
        text = recognizer.recognize_google(audio_data, language="ru-RU")
    return text


# Функция для переписки текста согласно правилам русского языка и расстановки пунктуации в тексте
def rewrite_text(text):
    model, example_texts, languages, punct, apply_te = torch.hub.load(
        repo_or_dir='snakers4/silero-models', model='silero_te')
    result = apply_te(text, lan='ru')
    return result


def get_speech_rewrited(result):
    # Проверяем, установлен ли ресурс punkt
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Если нет, то скачиваем его
        nltk.download('punkt')
    # Вызываем функции и выводим результаты
    result = rewrite_text(result)  # Переписываем текст
    return result


class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder=None):
        super().__init__(master)

        if placeholder is not None:
            self.placeholder = placeholder
            self.placeholder_color = 'grey'
            self.default_fg_color = self['fg']

            self.bind("<FocusIn>", self.focus_in)
            self.bind("<FocusOut>", self.focus_out)

            self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def focus_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def focus_out(self, *args):
        if not self.get():
            self.put_placeholder()


# Создаем окно приложения
window = tk.Tk()
window.title("Распознавание речи")
window.geometry("600x400")

# Создаем виджет для ввода ссылки на файл
link_entry = EntryWithPlaceholder(window, "Введите ссылку на файл")
# link_entry.insert(0, "Введите ссылку на файл")
link_entry.pack()

# Создаем виджет для выбора способа анализа аудио
analysis_var = tk.StringVar(window)
analysis_var.set("Выберите способ анализа аудио")
analysis_options = ["Vosk", "Google Speech Recognition"]
analysis_menu = tk.OptionMenu(window, analysis_var, *analysis_options)
analysis_menu.pack()

# Создаем виджет для индикатора выполнения
progress = ttk.Progressbar(window, mode="indeterminate")
progress.pack()


# Создаем функцию для обработки нажатия на кнопку старта компиляции
def start_compilation():
    # Создаем объект Thread с аргументом target=start_compilation
    thread = threading.Thread(target=start_compilation_in_thread)
    progress.start()
    # Вызываем метод start у объекта Thread
    thread.start()


# Создаем функцию для выполнения в отдельном потоке
def start_compilation_in_thread():
    # Получаем ссылку на файл из виджета ввода
    link = link_entry.get()
    # Получаем способ анализа аудио из виджета выбора
    analysis = analysis_var.get()
    # Проверяем, что ссылка и способ анализа не пустые
    if link and analysis:
        # Вызываем функцию для получения аудиофайла по ссылке
        audio_file = get_audio_file(link)
        # Если выбран способ анализа Vosk
        if analysis == "Vosk":
            # Вызываем функцию для распознавания речи с помощью Vosk
            result = recognize_speech(audio_file)
            result = get_speech_rewrited(result)
        # Если выбран способ анализа Google Speech Recognition
        else:
            # Вызываем функцию для распознавания речи с помощью Google Speech Recognition API
            result = recognize_speech_google(audio_file)
        # Выводим результат распознавания в виджет текста
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, result)
    else:
        # Выводим сообщение об ошибке в виджет текста
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Пожалуйста, введите ссылку на файл и выберите способ анализа аудио")
    progress.stop()


# Создаем виджет для кнопки старта компиляции
start_button = tk.Button(window, text="Начать компиляцию", command=start_compilation)
start_button.pack()

# Создаем виджет для вывода текста
text_widget = tk.Text(window)
text_widget.pack()

# Запускаем цикл обработки событий окна
window.mainloop()

# Создаем виджет для кнопки старта компиляции
start_button = tk.Button(window, text="Начать компиляцию", command=start_compilation)
start_button.pack()
