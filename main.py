import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# Mini Project 1 summer 2026 
# Nafisa Islam (40209761), Thajanah Mailvaganam (40114270), Zaree C Hameed (21026488)

# Q1: Load knowledge base CSV
# Knowledge base used to answer student questions. 

def load_knowledge_base(filename):
    try:
        df = pd.read_csv(filename)

        questions = df["question"].tolist()
        answers = df["answer"].tolist()

        return questions, answers

    except FileNotFoundError:
        print("Error: knowledge_base.csv file was not found.")
        print("Make sure the CSV file is in the same folder as this Python file.")
        exit()

    except Exception as error:
        print("Error loading knowledge base:", error)
        exit()



# Q2: Create embeddings
# This allows us to compare user questions with the knowledge base questions using cosine similarity.

def create_embeddings(model, questions):
    embeddings = model.encode(questions)
    return embeddings


# Q3: Sentiment Analysis
# This helps us understand the emotional tone of the user's question.

def sentiment_detector(sentiment_model, user_question):
    result = sentiment_model(user_question)[0]

    label = result["label"]
    confidence = result["score"]

    # This model gives labels as LABEL_0, LABEL_1, LABEL_2
    if label == "LABEL_0":
        sentiment_label = "NEGATIVE"
    elif label == "LABEL_1":
        sentiment_label = "NEUTRAL"
    elif label == "LABEL_2":
        sentiment_label = "POSITIVE"
    else:
        sentiment_label = label.upper()

    return sentiment_label, confidence



# Q4: Semantic Search
# It compares the user's question embedding with the knowledge base question.
# Returns similarity score.
def find_best_answer(user_question, model, question_embeddings, questions, answers):
    user_question_embedding = model.encode([user_question])

    comparison = cosine_similarity(user_question_embedding, question_embeddings)

    best_match_index = np.argmax(comparison)

    best_score = comparison[0][best_match_index]

    best_question = questions[best_match_index]
    best_answer = answers[best_match_index]

    return best_question, best_answer, best_score




def main():
    print("Loading Student Support AI...")

  
    questions, answers = load_knowledge_base("knowledge_base.csv")

    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings for knowledge base questions
    question_embeddings = create_embeddings(embedding_model, questions)

    #  for sentiment score
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    print("\nWelcome to Student Support AI")
    print("Type 'quit' to exit.")

    while True:
        user_question = input("\nHello :) How can I help you today? ").strip()

        
        if user_question.lower() == "quit":
            print("Goodbye!")
            break

        
        if user_question == "":
            print("Please enter a question.")
            continue

        # This helps detect the sentiment 
        sentiment_label, confidence = sentiment_detector(sentiment_model, user_question)

        print(f"\nSentiment: {sentiment_label} ({confidence:.2f})")

        # if it's negative sentiment
        if sentiment_label == "NEGATIVE" and confidence > 0.90:
            print("Recommended escalation: Contact human advisor.")

        # Find closest answer
        best_question, best_answer, best_score = find_best_answer(
            user_question,
            embedding_model,
            question_embeddings,
            questions,
            answers
        )

        print(f"Closest question: {best_question}")
        print(f"Similarity score: {best_score:.2f}")
        print(f"Answer: {best_answer}")


# Run run run run run the program
main()