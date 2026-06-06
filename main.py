import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# Mini Project 1 summer 2026  
# Nafisa Islam (40209761), Thajanah Mailvaganam (40114270), Zaree C Hameed (21026488)

 
def getknowledgeBase(filename):
    """
    Loads the knowledge base from a CSV file.

    Parameters: 
        filename (str) : the name of the csv file 

    Returns:a tuple of two lists: questions and answers
    """
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



def convert_questions_to_embeddings(model, questions):
    """
    Takes the knowledge base questions to make the embeddings ( numberical representations)
    Parameters: 
        model (SentenceTransformer) : the sentence transformer model to create the embeddings
        questions (list) : list of questions from the knowledge base
    Returns: a list of embeddings for the questions
    """
    embeddings = model.encode(questions)
    return embeddings


def getSentiment(sentiment, user_question):

    """
    Analyzes the sentiment of the user's question
    Parameters: 
        sentiment (pipeline) : the sentiment analysis model
        user_question (str) : the user's question to analyze
    Returns: a tuple of the sentiment ( NEGATIVE, NEUTRAL, or POSITIVE) and confidence score (0-100%)
    """
    result = sentiment(user_question)[0]

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




def getBestAnser(user_question, model, question_embeddings, questions, answers):
    """
    Finds the best matching question and answer from the knowledge base based on cosine similarity.
    Parameters:
    user_question (str) : the user's question
    model (SentenceTransformer) : the sentence transformer model takes the user's input/question and converts it into an embedding (numerical)
    question_embeddings (list) : list of embeddings for the knowledge base questions
    Returns: a tuple of the best matching question, its corresponding answer, and the similarity score
    """
    
    user_question_embedding = model.encode([user_question])

    comparison = cosine_similarity(user_question_embedding, question_embeddings)

    best_match_index = np.argmax(comparison)

    best_score = comparison[0][best_match_index]

    best_question = questions[best_match_index]
    
    if best_score < 0.60:

        best_answer = "Please contact a human advisor for further guidance."

    else:

        best_answer = answers[best_match_index]

    return best_question, best_answer, best_score




def main():
    print("Welcome to the Student Support AI! ")

  
    questions, answers = getknowledgeBase("knowledge_base.csv")

    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings for knowledge base questions
    question_embeddings = convert_questions_to_embeddings(embedding_model, questions)

    #  for sentiment score
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )


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
        sentiment_label, confidence = getSentiment(sentiment, user_question)

        print(f"\nSentiment: {sentiment_label} ({confidence:.2f})")

        # if it's negative sentiment
        if sentiment_label == "NEGATIVE" and confidence > 0.90:
            print("Please contact a human advisor for further guidance.")
            continue

        # Find closest answer
        best_question, best_answer, best_score = getBestAnser(
            user_question,
            embedding_model,
            question_embeddings,
            questions,
            answers
        )

        # print(f"Closest question: {best_question}")
        print(f"Similarity score: {best_score:.2f}")
        print(f"Answer: {best_answer}")


# Run run run run run the program
main()