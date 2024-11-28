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

# Knowledge Base
ipc_sections = [
    # Crimes Against the Human Body
    {"section": "320B", "keywords": "grievous hurt injuries serious harm damage injury wound assault bruise fracture disfigurement impairment permanent disability maim trauma bleeding incapacitation violent act body harm"},
    {"section": "302", "keywords": "murder homicide kill premeditated manslaughter assassination execution slaughter massacre crime death intentional fatality violence murder plan contract killing deadly act grievous death intentional killing malicious intent"},
    {"section": "307", "keywords": "attempted murder harm attack life danger assault premeditated lethal violent stabbing shooting poisoning strangulation deadly force intention to kill attempted homicide malicious attack violent crime assassination plot"},
    {"section": "304B", "keywords": "dowry death bride harassment family violence unnatural death cruelty abuse domestic violence dowry demands unnatural circumstances coercion marital abuse dowry victim bride killing family oppression"},
    {"section": "376", "keywords": "rape sexual assault force consent harassment molestation abuse violation intimacy misconduct coercion exploitation gender violence penetration indecency improper touch outraging modesty coercive sex exploitation degrading act"},
    {"section": "323", "keywords": "voluntarily causing hurt physical harm slap punch kick bruises injury harassment beating minor injuries force violence abuse brawl scuffle bodily pain hurt"},
    {"section": "341", "keywords": "wrongful restraint blocking path obstruction freedom unlawful detention barricade prevent passage restrict movement physical restraint holding captive confinement"},
    {"section": "498A", "keywords": "domestic violence harassment dowry cruelty abuse torture coercion maltreatment oppression mental anguish extortion humiliation marital discord partner abuse spousal violence marital abuse domestic cruelty psychological abuse family conflict"},
    {"section": "312", "keywords": "abortion illegal pregnancy terminate miscarriage prenatal fetus medical unsafe induced unwanted procedure contraception failure unlawful surgery birth control health issues unborn child termination"},

    # Crimes Against Property
    {"section": "379", "keywords": "theft steal property criminal robbery pilfering shoplifting embezzlement burglary larceny heist misappropriation possession snatch pickpocketing unauthorized access fraud trickery swindle unlawful taking"},
    {"section": "420", "keywords": "fraud cheating financial scam deception forgery trickery swindle counterfeit misrepresentation manipulation corruption embezzlement fraudster dishonesty money laundering breach of trust fraudulent transactions false promise"},
    {"section": "454", "keywords": "burglary break-in trespassing theft intrusion unauthorized entry housebreaking robbery unlawful occupation ransack breaking doors window smashing looting private property breach illegal access dwelling invasion unauthorized stay"},
    {"section": "406", "keywords": "criminal breach of trust embezzlement fraud misappropriation trust cheating fiduciary betrayal dishonesty unlawful possession breach deceit dishonored trust"},

    # Offenses Against Public Tranquility
    {"section": "141", "keywords": "unlawful assembly riot mob violence disruption disturbance public order illegal gathering criminal intent commotion group disorder hostility breach of peace congregation illegal"},
    {"section": "146", "keywords": "rioting violence mob attack disruption public safety disorder law and order brawl public disturbance unlawful assembly group conflict crowd chaos uproar rebellion civil unrest"},
    {"section": "144", "keywords": "prohibition assembly unlawful gathering curfew public safety restriction group order emergency prohibitory orders"},

    # Offenses Relating to Religion
    {"section": "295A", "keywords": "deliberate insult religious feelings outrage hurt sentiments blasphemy disrespect insult religion offense hate speech sacrilege community disruption communal disharmony religious insult religious provocation"},

    # Criminal Intimidation, Insult, and Annoyance
    {"section": "506", "keywords": "threaten kill harm death violence abuse intimidate menace extort blackmail coercion fear terrorize harassment scare insult provoke force unlawful demand warning retaliation aggression hostility"},
    {"section": "509", "keywords": "insult modesty gesture sexual harassment abuse eve-teasing verbal abuse obscene words disrespect harassment obscene remarks intimidation inappropriate behavior offensive gestures public shame defamation"},
    
    # Miscellaneous Offenses
    {"section": "363", "keywords": "kidnap abduct missing person hostage unlawful confinement lure detain seize disappear trafficking child snatch force captivity imprisonment ransom runaway unlawful restraint coercion abduction abduction cases"},
    {"section": "270", "keywords": "malignant act disease infection spread epidemic biological harm public health contamination biohazard endangerment negligence disease outbreak unsafe behavior infection control"},
    {"section": "279", "keywords": "rash driving public safety reckless vehicle danger speed accident traffic rules safety violation careless driving endangerment negligence hit-and-run"},
    {"section": "304A", "keywords": "causing death negligence accidental death carelessness recklessness unsafe practices unintentional killing negligence liability road accident industrial accident workplace safety oversight fault unintentional crime"},
    {"section": "186", "keywords": "obstructing public servant duty interference disruption prevention police official authority lawful action hindrance resistance authority government worker obstruction official work"},
    # General Explanations
    {"section": "34", "keywords": "common intention criminal act shared purpose joint liability group crime collective responsibility conspiracy cooperation group intent"},
    {"section": "120B", "keywords": "criminal conspiracy agreement unlawful plan intent collusion plotting scheming crime planning illegal association joint intent conspiracy crime"},
    # Offenses Against the State
    {"section": "121", "keywords": "waging war against government state rebellion treason sedition insurgency armed uprising revolution state threat treachery"},
    {"section": "124A", "keywords": "sedition disaffection government rebellion criticism incite violence speech hate speech overthrow criticism subversion dissent unlawful intent"},
    # Offenses Relating to the Army
    {"section": "131", "keywords": "abetting mutiny armed forces army rebellion sedition military insubordination disobedience war unlawful activity military misconduct"},
    # Offenses Relating to Marriage
    {"section": "494", "keywords": "bigamy second marriage unlawful marriage polygamy adultery unlawful spouse relationship multiple spouses second wedding unauthorized marriage criminal marital fraud"},
    #other crimes
    {"section": "511", "keywords": "attempt to commit offense unsuccessful crime incomplete act preparation conspiracy attempt criminal intent unexecuted crime effort to commit"},
    {"section": "279", "keywords": "rash driving reckless driving vehicle accident public safety traffic rules violation endangerment speeding hit and run carelessness negligence"},
    {"section": "186", "keywords": "obstructing public servant duty prevention interference resistance police officer disruption lawful authority hindrance refusal disobedience unlawful act"}    

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
