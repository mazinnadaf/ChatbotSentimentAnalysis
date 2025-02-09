#sentiment analysis, find a use for that intent
import torch
import spacy
from abc import abstractmethod, ABCMeta
from openai import AssistantEventHandler, OpenAI
import json
import sqlite3
from termcolor import colored
from intent_model import load_trained_model,train_model
from datetime import datetime
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import nltk
nltk.download('punkt')

# Get current timestamp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
#from new_intent_model import new_sentiment_analysis_with_lstm
from intent_mdl_inference import new_sentiment_analysis_with_lstm
from intent_mdl_inference import *
from senty_mdl_inf_svd_mdl import *
from sentiment_inf import SentimentInference

def setup_database():
    conn = sqlite3.connect('chatbot_memory.db')
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        message TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')

    conn.commit()
    return conn, cursor

def save_conversation(cursor, user_id, message, response,current_timestamp):
    cursor.execute('''
    INSERT INTO conversations (user_id, message, response,timestamp)
    VALUES (?, ?, ?,?)
    ''', (user_id, message, response,current_timestamp))
    conn.commit()

def get_conversations(cursor, user_id):
    cursor.execute('''
    SELECT message, response,timestamp FROM conversations
    WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100
    ''', (user_id,))

    fetched_history=cursor.fetchall()
    df = pd.DataFrame(fetched_history, columns=['User Input', 'Bot Response', 'Timestamp'])

    conversation_history = ""

    for conv in fetched_history:
        conversation_history += f"User: {conv[0]}\nBot: {conv[1]}\nTimestamp:{conv[2]}\n"
       # print(conversation_history)

    return df

def get_counts(cursor, user_id):
    cursor.execute('''
    SELECT count(*) FROM conversations
    ''')
    print(" counts")
    print(cursor.fetchall())
    return cursor.fetchall()


def clear_database():
    conn = sqlite3.connect('chatbot_memory.db')
    cursor = conn.cursor()

    cursor.execute('DELETE FROM conversations')
    cursor.execute('DELETE FROM users')

    conn.commit()
    conn.close()

    print("Database has been cleared.")

class BotPrompter(metaclass=ABCMeta):
    @abstractmethod
    def __str__(self):
        pass

class GeneralPrompter(BotPrompter):
    def __init__(self, gen_stmt):
        self.gen_query = gen_stmt

    def __str__(self):
        return f'Please generate response for this query: {self.gen_query}' \
               f' DO NOT return any other words before or after that paragraph.' \
               f' DO NOT use ellipses in that paragraph.'

class LlmPOCBot:
    def __init__(self, model='gpt-3.5-turbo'):
        self.model = model
        self.client = OpenAI(
            api_key="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")
        self.COMPLETIONS_MODEL = "gpt-3.5-turbo"

    def message(self, message):
        completion = self.client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': str(message)}])
        return completion.choices[0].message.content

    def query(self, prompter: BotPrompter, user_id: int):
        full_prompt = f"User: {prompter.gen_query}\nBot:"

        return self.message(full_prompt)
        #return self.message(str(prompter))

conn, cursor = setup_database()
generator = LlmPOCBot()

user_id = 1
message = ""

##sentiment analysis and Data labeling with "sentiment column for old chat history" for ML mode

recent_conversations = get_conversations(cursor, 1)
nltk.download('vader_lexicon')
#chat_history = pd.DataFrame(recent_conversations)
chat_history = pd.DataFrame(recent_conversations)

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
def get_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

chat_history["sentiment"] = chat_history["User Input"].apply(get_sentiment)

print(chat_history.columns)

## Call the LSTM Model on Chat history with labelled data to train and save the model
print("counts")
print(get_counts)

sentiment_counts = chat_history["sentiment"].value_counts()
print(sentiment_counts)
train_and_save_model(chat_history)

### load the saved tranied model###
inference = SentimentInference(model_path='sentiment_model.pth')

while message != 'quit':
    message = str(input(colored("***Enter message or type 'quit' to exit***\n", 'blue', 'on_white')))

    if message != 'quit':

        prompter = GeneralPrompter(message)
        response = generator.query(prompter, user_id)

        current_timestamp = datetime.now()

        save_conversation(cursor, user_id, message, response,current_timestamp)
        conn.commit()
        ##### run inference on each chat message#########
        #predict_fn = new_sentiment_analysis_with_lstm(df)

        # Predict sentiment for new sentences
        new_sentences = [message]
        #predicted_labels = predict_new_sentences(new_sentences)
        #print(predicted_labels)
        predictions = inference.predict(new_sentences)
        print(predictions)
        ################################

        print(colored(response, 'blue', 'on_yellow'))
        print(colored("                                                                                                                                     ",'blue','on_white'))

conn.close()