from email.mime import text

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob

# Q3 Sentiment Analysis
def sentiment_detector(question):

    analysis = TextBlob(question)
    sentiments= analysis.sentiment.polarity

    if sentiments > 0:
        return "Positive"

    elif sentiments < 0:
        return "Negative"

    else:
        return "Neutral"

df = pd.read_csv('knowledge_base.csv')


# Q1 csv question
questions = df["question"].tolist()
answers = df["answer"].tolist()

# 2.1 choosing AI model for embedding the most simple one
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddingQuestions = model.encode(questions)
embeddingAnswers = model.encode(answers)   

 # Q2.2 Compare user queries to stored questions 
while True:

    yourQuestion = input("\nWhat is your question? :  ")

    if yourQuestion == "exit":
        break
    
    sentiment = sentiment_detector(yourQuestion)

    print(f"\nSentiment: {sentiment}")
    
    yourQuestionEmbedding = model.encode([yourQuestion]) 

    comparison = cosine_similarity(yourQuestionEmbedding, embeddingQuestions)


    # Q2.3 Return the most similar question and answer
    
    bestMatch_index = np.argmax(comparison)

    bestMatch = comparison[0][bestMatch_index]

    print("\nAnswer: " + answers[bestMatch_index])
   




   