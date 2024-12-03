import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

ipc_sections = [
    # Section 1: Preliminary
    {"section": "1", "keywords": "introduction scope application definition interpretation preliminary provisions legal framework act commencement"},

    # Section 2-4: Geographical Jurisdiction
    {"section": "2", "keywords": "territorial jurisdiction india indian territory applicability geographical extent legal boundaries territorial limits"},
    {"section": "3", "keywords": "definition explanations interpretation legal terms words phrases meaning"},
    {"section": "4", "keywords": "extension of act territorial limits extraterritorial jurisdiction indian penal code application"},

    # Section 5-23: General Exceptions
    {"section": "5", "keywords": "act omission different character intention circumstances"},
    {"section": "6", "keywords": "person causing act law permitting justification legal authorization"},
    {"section": "7", "keywords": "criminal act omission defined punishment explanation"},
    {"section": "8", "keywords": "act done intention different effect actual consequence"},
    {"section": "9", "keywords": "hatred malice intention wrong"},
    {"section": "10", "keywords": "intention effect joint acts consequences shared liability"},
    {"section": "11", "keywords": "false charge malicious prosecution wrongful accusation"},
    {"section": "12", "keywords": "offence committed by person of unsound mind mental health"},
    {"section": "13", "keywords": "child under seven immature age responsibility"},
    {"section": "14", "keywords": "child above seven under twelve limited responsibility"},
    {"section": "15", "keywords": "immature offender limited culpability"},
    {"section": "16", "keywords": "act done by consent valid legal permission"},
    {"section": "17", "keywords": "consent communication agreement permission"},
    {"section": "18", "keywords": "communication verbal non-verbal express implied"},
    {"section": "19", "keywords": "consent withdrawal revocation cancellation"},
    {"section": "20", "keywords": "consent elements validity requirements"},
    {"section": "21", "keywords": "communication mistake error misunderstanding"},
    {"section": "22", "keywords": "intoxication alcohol drugs mental state"},
    {"section": "23", "keywords": "right private defense protection self-defense"},

    # Offences against the State
    {"section": "121", "keywords": "waging war against india treason national security sedition rebellion armed conflict"},
    {"section": "121A", "keywords": "conspiracy to wage war against india seditious planning national threat"},
    {"section": "122", "keywords": "collecting arms to wage war against india weapons accumulation"},
    {"section": "123", "keywords": "concealing design to wage war national security threat"},

    # Offences against Public Tranquility
    {"section": "141", "keywords": "unlawful assembly riot public disturbance collective action"},
    {"section": "142", "keywords": "being member of unlawful assembly participation"},
    {"section": "143", "keywords": "punishment for unlawful assembly"},
    {"section": "144", "keywords": "joining unlawful assembly armed with deadly weapon"},
    {"section": "145", "keywords": "joining or continuing in unlawful assembly"},

    # Murder and Culpable Homicide
    {"section": "299", "keywords": "culpable homicide definition killing intentional knowledge recklessness"},
    {"section": "300", "keywords": "murder intentional killing deliberate homicide premeditated death"},
    {"section": "301", "keywords": "punishment for murder death sentence capital punishment"},
    {"section": "302", "keywords": "punishment for murder life imprisonment death penalty"},
    {"section": "303", "keywords": "murder by life convict death sentence"},
    {"section": "304", "keywords": "punishment for culpable homicide not amounting to murder"},

    # Hurt and Injury
    {"section": "319", "keywords": "hurt bodily pain injury suffering physical harm"},
    {"section": "320", "keywords": "grievous hurt serious bodily injury permanent damage disfigurement"},
    {"section": "321", "keywords": "voluntarily causing hurt assault battery"},
    {"section": "322", "keywords": "voluntarily causing grievous hurt dangerous weapon"},
    {"section": "323", "keywords": "voluntarily causing hurt"},
    {"section": "324", "keywords": "voluntarily causing hurt with dangerous weapon"},
    {"section": "325", "keywords": "voluntarily causing grievous hurt"},
    {"section": "326", "keywords": "voluntarily causing grievous hurt with dangerous weapon"},

    # Sexual Offences
    {"section": "375", "keywords": "rape sexual assault non-consensual intercourse sexual violence"},
    {"section": "376", "keywords": "punishment for rape sexual assault imprisonment"},
    {"section": "354", "keywords": "assault outraging modesty of woman sexual harassment"},
    {"section": "509", "keywords": "word gesture insult modesty of woman"},

    # Theft and Robbery
    {"section": "378", "keywords": "theft stealing taking property without consent movable property"},
    {"section": "379", "keywords": "punishment for theft stealing"},
    {"section": "380", "keywords": "theft in dwelling house"},
    {"section": "381", "keywords": "theft by clerk or servant"},
    {"section": "382", "keywords": "preparation for theft"},
    {"section": "392", "keywords": "robbery theft with violence force intimidation"},
    {"section": "393", "keywords": "attempt to commit robbery"},

    # Cheating and Fraud
    {"section": "415", "keywords": "cheating deception fraud dishonest inducement"},
    {"section": "416", "keywords": "cheating by personation false identity"},
    {"section": "417", "keywords": "punishment for cheating"},
    {"section": "418", "keywords": "cheating with knowledge of probability of injury"},
    {"section": "419", "keywords": "punishment for cheating by personation"},
    {"section": "420", "keywords": "cheating and dishonestly inducing delivery of property"},

    # Criminal Breach of Trust
    {"section": "405", "keywords": "criminal breach of trust misappropriation property dishonesty"},
    {"section": "406", "keywords": "punishment for criminal breach of trust"},
    {"section": "407", "keywords": "criminal breach of trust by carrier"},
    {"section": "408", "keywords": "criminal breach of trust by clerk or servant"},
    {"section": "409", "keywords": "criminal breach of trust by public servant"},

    # Counterfeiting
    {"section": "231", "keywords": "counterfeiting government stamp forging official document"},
    {"section": "232", "keywords": "abetting counterfeiting of stamp"},
    {"section": "233", "keywords": "sale of counterfeit stamp"},
    {"section": "234", "keywords": "making or selling instrument for counterfeiting"},
    {"section": "235", "keywords": "counterfeiting device or mark used for authenticating documents"}
]

# Text Preprocessing 
def preprocess_text(text):
    text = text.lower()  
    words = text.split()  
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] 
    return " ".join(words)

# Preprocess the keywords
processed_ipc_keywords = [preprocess_text(ipc["keywords"]) for ipc in ipc_sections]

# Convert Knowledge Base to TF-IDF Vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  
ipc_vectors = vectorizer.fit_transform(processed_ipc_keywords)

# Define fuzzy variables for similarity score
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
rule2 = ctrl.Rule(score["medium"], membership["applicable"])
rule3 = ctrl.Rule(score["high"], membership["applicable"])

# Create control system
membership_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
membership_simulation = ctrl.ControlSystemSimulation(membership_ctrl)

# Function to analyze a complaint
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
