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

# Step 1: Expanded Knowledge Base (IPC Sections with Keywords)
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
    {"section": "302", "keywords": "murder homicide kill"},
    # Add more sections as needed...
]

# Step 2: Preprocess Text Function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    words = text.split()  # Split text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

# Preprocess the keywords for IPC sections
processed_ipc_keywords = [preprocess_text(ipc["keywords"]) for ipc in ipc_sections]

# Step 3: TF-IDF Vectorizer Setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_ipc_keywords)  # Fit on IPC sections only

# Step 4: Fuzzification Setup
score = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "score")
membership = ctrl.Consequent(np.arange(0, 1.1, 0.1), "membership")

# Membership functions
score["low"] = fuzz.trimf(score.universe, [0, 0, 0.2])
score["medium"] = fuzz.trimf(score.universe, [0, 0.2, 0.5])
score["high"] = fuzz.trimf(score.universe, [0.5, 1, 1])
membership["not_applicable"] = fuzz.trimf(membership.universe, [0, 0, 0.5])
membership["applicable"] = fuzz.trimf(membership.universe, [0.5, 1, 1])

# Fuzzy rules
rule1 = ctrl.Rule(score["low"], membership["not_applicable"])
rule2 = ctrl.Rule(score["medium"], membership["not_applicable"])
rule3 = ctrl.Rule(score["high"], membership["applicable"])

# Create control system
membership_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
membership_simulation = ctrl.ControlSystemSimulation(membership_ctrl)

# Step 5: Analyze Complaint
def analyze_complaint(complaint):
    # Preprocess the complaint
    processed_complaint = preprocess_text(complaint)
    
    # Vectorize the complaint
    complaint_vector = vectorizer.transform([processed_complaint])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(complaint_vector, tfidf_matrix).flatten()
    
    # Calculate fuzzy membership values and gather applicable IPC sections
    applicable_sections = []
    for idx, score_value in enumerate(similarity_scores):
        membership_simulation.input["score"] = score_value
        membership_simulation.compute()
        fuzzy_value = membership_simulation.output["membership"]
        if fuzzy_value > 0.1:  # Threshold for considering the section
            applicable_sections.append({
                "section": ipc_sections[idx]["section"],
                "similarity": round(score_value, 4),
                "fuzzy_score": round(fuzzy_value, 4),
            })
    
    # Sort by fuzzy membership score in descending order
    applicable_sections = sorted(applicable_sections, key=lambda x: x["fuzzy_score"], reverse=True)
    return applicable_sections

# Step 6: Generate HTML Report
def generate_html_report(complaint, analysis_results):
    html_content = f"""
    <html>
    <head>
        <title>Complaint Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                color: #333;
            }}
            h1 {{
                text-align: center;
                color: #003366;
            }}
            table {{
                width: 80%;
                margin: 20px auto;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #003366;
                color: white;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Complaint Analysis</h1>
        <p><strong>Complaint:</strong> {complaint}</p>
        <table>
            <tr>
                <th>IPC Section</th>
                <th>Similarity Score</th>
                <th>Fuzzy Membership</th>
            </tr>
    """
    for result in analysis_results:
        html_content += f"""
            <tr>
                <td>{result['section']}</td>
                <td>{result['similarity']}</td>
                <td>{result['fuzzy_score']}</td>
            </tr>
        """
    html_content += """
        </table>
    </body>
    </html>
    """
    # Save to file
    with open("complaint_analysis.html", "w") as file:
        file.write(html_content)
    print("HTML report generated successfully!")

# Step 7: Take Complaint as Input and Analyze
complaint_input = input("Enter the complaint: ")
results = analyze_complaint(complaint_input)

# Generate HTML Report
generate_html_report(complaint_input, results)

# Display results in the console
if results:
    print("Applicable IPC Sections:")
    for res in results:
        print(f"IPC {res['section']}: Similarity={res['similarity']}, Fuzzy Score={res['fuzzy_score']}")
else:
    print("No applicable IPC sections found.")
