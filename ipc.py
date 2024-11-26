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
    {"section": "1", "keywords": "India territorial jurisdiction application law"},
    {"section": "2", "keywords": "punishment definition criminal offence"},
    {"section": "3", "keywords": "punishment alternative punishment offence"},
    {"section": "4", "keywords": "special law overriding general law"},
    {"section": "6", "keywords": "extent jurisdictional limit territory"},
    {"section": "7", "keywords": "criminal act definition legal offence"},
    {"section": "8", "keywords": "act committed partly within territory"},
    {"section": "9", "keywords": "punishment extension legal provisions"},
    {"section": "10", "keywords": "connection separate offences joint trial"},
    {"section": "11", "keywords": "definition legal terminology explanation"},
    {"section": "12", "keywords": "digital electronic record evidence"},
    {"section": "13", "keywords": "public servant definition roles"},
    {"section": "14", "keywords": "government servant official duty"},
    {"section": "15", "keywords": "legal definition government servant"},
    {"section": "16", "keywords": "punishment criminal offence intent"},
    {"section": "17", "keywords": "culpable knowledge criminal intent"},
    {"section": "18", "keywords": "voluntary act criminal intention"},
    {"section": "19", "keywords": "intent criminal action definition"},
    {"section": "20", "keywords": "wrong malicious intent criminal"},
    {"section": "21", "keywords": "harm injury consequence action"},
    {"section": "22", "keywords": "intoxication criminal responsibility"},
    {"section": "23", "keywords": "illegal act criminal intention"},
    {"section": "24", "keywords": "dishonest wrongful gain intent"},
    {"section": "25", "keywords": "arms weapons illegal possession"},
    {"section": "26", "keywords": "land revenue document forgery"},
    {"section": "27", "keywords": "punishment lesser offence"},
    {"section": "28", "keywords": "attempt complete offence"},
    {"section": "29", "keywords": "offence part done completion"},
    {"section": "30", "keywords": "joint offence several persons"},
    {"section": "34", "keywords": "common intention joint action"},
    {"section": "35", "keywords": "criminal act several persons"},
    {"section": "37", "keywords": "co-operation criminal act"},
    {"section": "38", "keywords": "persons responsible criminal act"},
    {"section": "39", "keywords": "private person definition"},
    {"section": "40", "keywords": "injury definition legal term"},
    {"section": "41", "keywords": "private defence protection"},
    {"section": "42", "keywords": "limitation private defence"},
    {"section": "45", "keywords": "good faith legal protection"},
    {"section": "52", "keywords": "accident unavoidable circumstances"},
    {"section": "54", "keywords": "right private defence"},
    {"section": "76", "keywords": "mistake fact legal defence"},
    {"section": "79", "keywords": "accident inevitable circumstances"},
    {"section": "81", "keywords": "act prevent greater harm"},
    {"section": "82", "keywords": "child below seven criminal responsibility"},
    {"section": "84", "keywords": "unsound mind mental disorder"},
    {"section": "86", "keywords": "intoxication involuntary criminal act"},
    {"section": "87", "keywords": "consent bodily harm"},
    {"section": "88", "keywords": "medical treatment good faith"},
    {"section": "89", "keywords": "consent guardian minor"},
    {"section": "90", "keywords": "consent definition legal"}
]

# Step 2: Expanded Crime Reports (Multiple Complaints)
crime_reports = [
    "I was returning home on a bicycle when X attacked me causing serious injuries to my eyes and threatened to kill me.",
    "On 15/01/2023, my neighbor asked for a money loan. When I refused, he abused and threatened to beat me.",
    "Today my son was kidnapped while returning from school. The kidnapper demanded 2 lakh rupees and threatened to kill him.",
    "While admitted to the hospital, a ward boy assaulted me and threatened my family to keep silent.",
    "After my marriage, my husband forced an abortion against my will, causing me severe mental distress.",
    "My car was stolen from the parking lot. I was shocked to find it missing. The police suspect it was stolen.",
    "I have been continuously harassed by my husband and his family for dowry. They constantly threaten me with physical harm.",
    "I was duped by an online shopping fraud where I paid for goods but never received the items.",
    "A man tried to kill me with a knife after I rejected his advances. He attempted to stab me multiple times.",
    "I found my house broken into, and many valuables were stolen. The thief had broken the lock and entered through the back door.",
    "A man was caught murdering his wife after a long history of violent arguments. He strangled her in their home."
]

# Step 3: Text Preprocessing - Remove stopwords, Lemmatize, and Combine keywords
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    words = text.split()  # Split text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)

# Preprocess the keywords and crime reports
processed_ipc_keywords = [preprocess_text(ipc["keywords"]) for ipc in ipc_sections]
processed_crime_reports = [preprocess_text(report) for report in crime_reports]

# Step 4: Convert Knowledge Base and Reports to TF-IDF Vectors
vectorizer = TfidfVectorizer()
all_texts = processed_ipc_keywords + processed_crime_reports
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Separate IPC and reports vectors
ipc_vectors = tfidf_matrix[:len(ipc_sections)]
crime_vectors = tfidf_matrix[len(ipc_sections):]

# Step 5: Compute Similarity Scores
similarity_scores = cosine_similarity(crime_vectors, ipc_vectors)

# Step 6: Fuzzification - Define fuzzy variables for similarity score
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

# Step 7: Calculate Fuzzy Membership for Each IPC Section per Report
fuzzy_results = []
html_content = """
<html>
<head>
    <title>Crime Report Analysis - Government of India</title>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            background-color: #f4f4f9;
            color: #333;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-top: 20px;
            color: #003366;
            font-weight: bold;
        }
        .header {
            background-color: #003366;
            color: white;
            padding: 10px 0;
            font-size: 1.5em;
            text-align: center;
        }
        table {
            width: 80%;
            margin: 20px auto;
            border-collapse: collapse;
            border: 1px solid #ddd;
            font-size: 1.1em;
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        th {
            background-color: #003366;
            color: white;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 20px;
            color: #888;
            background-color: #f4f4f9;
            padding: 10px;
        }
        .logo {
            max-width: 100px;
            display: block;
            margin: 0 auto;
        }
        .intro {
            font-size: 1.2em;
            margin-top: 20px;
            text-align: center;
            color: #003366;
        }
    </style>
</head>
<body>
    <div class="header">
        <img class="logo" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Emblem_of_India.svg/1024px-Emblem_of_India.svg.png" alt="India Emblem">
        Crime Report Analysis - Government of India
    </div>
    <h1>Crime Report Analysis</h1>
    <div class="intro">
        <p>This document presents the analysis of various crime reports based on the Indian Penal Code (IPC). Each report is compared with relevant IPC sections, and the most applicable sections are identified based on the similarity of the report's content.</p>
    </div>
    <table>
        <tr>
            <th>Crime Report</th>
            <th>Similarity Scores</th>
            <th>Applicable IPC Sections</th>
        </tr>
"""

for report_idx, report_scores in enumerate(similarity_scores):
    applicable_sections = []
    similarity_list = []
    for section_idx, score_value in enumerate(report_scores):
        membership_simulation.input["score"] = score_value
        membership_simulation.compute()
        fuzzy_value = membership_simulation.output["membership"]
        
        # Adjust threshold to include sections with fuzzy value > 0.1
        if fuzzy_value > 0.1:
            applicable_sections.append(
                (ipc_sections[section_idx]["section"], fuzzy_value)
            )
            similarity_list.append(f"IPC {ipc_sections[section_idx]['section']} - Score: {score_value:.4f}, Fuzzy: {fuzzy_value:.4f}")
    
    # Create HTML table row for each report
    crime_report_html = f"<td>{crime_reports[report_idx]}</td>"
    similarity_html = f"<td>{', '.join(similarity_list) if similarity_list else 'No applicable sections'}</td>"
    applicable_html = f"<td>{', '.join([section[0] for section in applicable_sections]) if applicable_sections else 'None'}</td>"
    
    html_content += f"<tr>{crime_report_html}{similarity_html}{applicable_html}</tr>"

html_content += """
    </table>
    <div class="footer">
        <p>Crime Report Analysis - Indian Penal Code Section Prediction</p>
        <p>Data provided by the Government of India</p>
        <p>For further inquiries, please contact the Ministry of Law and Justice</p>
    </div>
</body>
</html>
"""

# Step 8: Save the HTML content to a file
with open("crime_report_analysis.html", "w") as file:
    file.write(html_content)

print("HTML report generated successfully!")
