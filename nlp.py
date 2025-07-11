import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from colorama import Fore, Style, init

init(autoreset=True)

# Load FAQ data 
with open("faq_data.json", "r") as file:
    qa_pairs = json.load(file)

questions = list(qa_pairs.keys())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chatbot with fuzzy matching
def chatbot_nlp_fuzzy():
    print(Fore.GREEN +"🤖 SevaBot: Hello! Ask your question (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").lower().strip()
        if user_input == "exit":
            print(Fore.BLUE +"🤖 SevaBot: Goodbye!")
            break

        query_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(query_vec, X)
        idx = similarity.argmax()
        tfidf_score = similarity[0][idx]

        fuzzy_matches = get_close_matches(user_input, questions, n=1, cutoff=0.6)

        if tfidf_score > 0.3 or fuzzy_matches:
            matched_question = (
                questions[idx] if tfidf_score >= 0.3 else fuzzy_matches[0]
            )
            print(Fore.YELLOW +"🤖 SevaBot:", qa_pairs[matched_question])
        else:
            print(Fore.GREEN +"🤖 SevaBot: Sorry, I didn’t understand. Please enter your query again.")

chatbot_nlp_fuzzy()
