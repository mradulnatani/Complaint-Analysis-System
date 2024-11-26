import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if not already available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Step 1: Knowledge Base (IPC Sections with Keywords)
ipc_sections = [
    {"section": "320B", "keywords": "grievous hurt injuries serious harm damage injury"},
    {"section": "506", "keywords": "threaten kill harm death violence abuse"},
    {"section": "363", "keywords": "kidnap abduct missing person"},
    {"section": "376D", "keywords": "rape sexual assault force consent"},
    {"section": "312", "keywords": "abortion illegal pregnancy terminate"},
    {"section": "379", "keywords": "theft steal property criminal"},
    {"section": "498A", "keywords": "domestic violence harassment dowry cruelty"},
    {"section": "420", "keywords": "fraud cheating financial scam"},
    {"section": "307", "keywords": "attempted murder harm attack"},
    {"section": "454", "keywords": "burglary break-in trespassing theft"},
    {"section": "302", "keywords": "murder homicide kill"}
]

# Step 2: Text Preprocessing - Remove stopwords, Lemmatize, and Combine keywords
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    words = text.split()  # Split text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

# Preprocess the keywords
processed_ipc_keywords = [preprocess_text(ipc["keywords"]) for ipc in ipc_sections]

# Step 3: Convert Knowledge Base to TF-IDF Vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use bigrams for better context understanding
ipc_vectors = vectorizer.fit_transform(processed_ipc_keywords)

# Step 4: Define fuzzy variables for similarity score
score = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "score")
membership = ctrl.Consequent(np.arange(0, 1.1, 0.1), "membership")

# Membership functions
score["low"] = fuzz.trimf(score.universe, [0, 0, 0.3])
score["medium"] = fuzz.trimf(score.universe, [0.2, 0.5, 0.8])
score["high"] = fuzz.trimf(score.universe, [0.7, 1, 1])
membership["not_applicable"] = fuzz.trimf(membership.universe, [0, 0, 0.4])
membership["applicable"] = fuzz.trimf(membership.universe, [0.3, 0.7, 1])

# Fuzzy rules
rule1 = ctrl.Rule(score["low"], membership["not_applicable"])
rule2 = ctrl.Rule(score["medium"], membership["not_applicable"])
rule3 = ctrl.Rule(score["high"], membership["applicable"])

# Create control system
membership_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
membership_simulation = ctrl.ControlSystemSimulation(membership_ctrl)

# Step 5: Function to analyze a complaint
def analyze_complaint(complaint_text):
    # Preprocess the input complaint
    processed_complaint = preprocess_text(complaint_text)
    complaint_vector = vectorizer.transform([processed_complaint])

    similarity_scores = cosine_similarity(complaint_vector, ipc_vectors)[0]


    print("Similarity Scores:", similarity_scores)

    applicable_sections = []
    similarity_list = []
    for section_idx, score_value in enumerate(similarity_scores):
        # Use similarity score directly in fuzzy input
        membership_simulation.input["score"] = score_value
        membership_simulation.compute()
        fuzzy_value = membership_simulation.output["membership"]

        # Debugging: Print fuzzy values
        print(f"Section {ipc_sections[section_idx]['section']}: Similarity={score_value:.4f}, Fuzzy={fuzzy_value:.4f}")

        # Adjust threshold to include more relevant sections
        if fuzzy_value > 0.1:  # Threshold to consider as applicable
            applicable_sections.append((ipc_sections[section_idx]["section"], fuzzy_value))
            similarity_list.append(
                f"IPC {ipc_sections[section_idx]['section']} - Score: {score_value:.4f}, Fuzzy: {fuzzy_value:.4f}"
            )

    if not applicable_sections:
        print("No IPC sections matched based on the current thresholds.")
    else:
        print(f"Applicable sections: {[sec[0] for sec in applicable_sections]}")

    return {
        "complaint": complaint_text,
        "applicable_sections": applicable_sections,
        "details": similarity_list,
    }

if __name__ == "__main__":
    complaint_input = input("Enter the complaint: ")
    results = analyze_complaint(complaint_input)
    
    print("\nComplaint Analysis Results:")
    print(f"Complaint: {results['complaint']}")
    print("\nApplicable IPC Sections:")
    for section, fuzzy_value in results["applicable_sections"]:
        print(f"  - IPC Section {section}: Fuzzy Score = {fuzzy_value:.4f}")
    print("\nDetails:")
    for detail in results["details"]:
        print(f"  {detail}")
