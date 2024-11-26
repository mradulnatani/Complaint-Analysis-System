# Complaint Analysis System with IPC Section Matching

## Overview
This program takes a user-submitted complaint and analyzes it by matching relevant Indian Penal Code (IPC) sections using Natural Language Processing (NLP) techniques. The system uses a combination of **TF-IDF vectorization**, **cosine similarity**, and **fuzzy logic** to determine the most applicable IPC sections based on the content of the complaint.

## Features
- **Text Preprocessing**: The complaint text is preprocessed by removing stopwords and applying lemmatization to ensure meaningful keyword extraction.
- **Keyword Matching**: IPC sections are associated with predefined keywords. These keywords are compared to the complaint text using the **TF-IDF** vectorizer to calculate similarity.
- **Fuzzy Logic**: Fuzzy logic is used to translate the cosine similarity scores into a membership value, which helps determine whether a particular IPC section is applicable to the complaint.
- **Threshold-based Filtering**: Only IPC sections with a fuzzy membership value above a threshold are considered relevant and returned as applicable.

## Dependencies
This project requires the following Python libraries:
- `pandas`: For handling data and manipulating tabular data.
- `numpy`: For numerical operations and arrays.
- `sklearn`: For machine learning utilities, including text vectorization and cosine similarity.
- `skfuzzy`: For fuzzy logic implementation and control systems.
- `nltk`: For natural language processing tasks, including tokenization, stopword removal, and lemmatization.

Install dependencies using:
```bash
pip install pandas numpy scikit-learn skfuzzy nltk
```

## Code Walkthrough

### 1. Knowledge Base (IPC Sections with Keywords)
The system defines a dictionary containing IPC sections and associated keywords. Each section is identified by a unique code (e.g., "320B", "506") and a set of keywords that represent the types of offenses related to the section.

```python
ipc_sections = [
    {"section": "320B", "keywords": "grievous hurt injuries serious harm damage injury"},
    {"section": "506", "keywords": "threaten kill harm death violence abuse"},
    ...
]
```

### 2. Text Preprocessing
The `preprocess_text` function processes the input complaint text and the keywords associated with each IPC section:
- Converts the text to lowercase.
- Splits the text into individual words.
- Removes stopwords (common words like "the", "and", etc.).
- Lemmatizes the remaining words to their root form.

```python
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    words = text.split()  # Split text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)
```

### 3. Vectorization of IPC Keywords
The IPC section keywords are processed using the `TfidfVectorizer` from **scikit-learn** to convert the keywords into numerical vectors. The vectors are calculated using **TF-IDF** (Term Frequency-Inverse Document Frequency) with unigrams and bigrams for better context understanding.

```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use bigrams for better context understanding
ipc_vectors = vectorizer.fit_transform(processed_ipc_keywords)
```

### 4. Fuzzy Logic Setup
Fuzzy logic is used to map similarity scores (from cosine similarity) to fuzzy membership values. Membership functions are defined for two variables:
- **score**: Represents the similarity score, divided into "low", "medium", and "high".
- **membership**: Represents the applicability of the IPC section, divided into "not applicable" and "applicable".

```python
score = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "score")
membership = ctrl.Consequent(np.arange(0, 1.1, 0.1), "membership")

score["low"] = fuzz.trimf(score.universe, [0, 0, 0.3])
score["medium"] = fuzz.trimf(score.universe, [0.2, 0.5, 0.8])
score["high"] = fuzz.trimf(score.universe, [0.7, 1, 1])

membership["not_applicable"] = fuzz.trimf(membership.universe, [0, 0, 0.4])
membership["applicable"] = fuzz.trimf(membership.universe, [0.3, 0.7, 1])
```

### 5. Fuzzy Rules
The fuzzy rules map similarity scores to applicability. If the score is "low" or "medium", the membership value will indicate that the IPC section is not applicable. If the score is "high", the IPC section is considered applicable.

```python
rule1 = ctrl.Rule(score["low"], membership["not_applicable"])
rule2 = ctrl.Rule(score["medium"], membership["not_applicable"])
rule3 = ctrl.Rule(score["high"], membership["applicable"])
```

### 6. Complaint Analysis Function
The main function, `analyze_complaint`, takes a complaint as input, preprocesses it, calculates its cosine similarity with each IPC section's keywords, and applies fuzzy logic to determine the applicability of each IPC section.

```python
def analyze_complaint(complaint_text):
    processed_complaint = preprocess_text(complaint_text)
    complaint_vector = vectorizer.transform([processed_complaint])

    similarity_scores = cosine_similarity(complaint_vector, ipc_vectors)[0]

    applicable_sections = []
    similarity_list = []
    for section_idx, score_value in enumerate(similarity_scores):
        membership_simulation.input["score"] = score_value
        membership_simulation.compute()
        fuzzy_value = membership_simulation.output["membership"]

        if fuzzy_value > 0.1:  # Threshold to consider as applicable
            applicable_sections.append((ipc_sections[section_idx]["section"], fuzzy_value))
            similarity_list.append(
                f"IPC {ipc_sections[section_idx]['section']} - Score: {score_value:.4f}, Fuzzy: {fuzzy_value:.4f}"
            )

    return {
        "complaint": complaint_text,
        "applicable_sections": applicable_sections,
        "details": similarity_list,
    }
```

### 7. Main Program Execution
The program prompts the user to input a complaint, then analyzes it and prints the results, including the applicable IPC sections and similarity details.

```python
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
```

### Sample Output
```
Enter the complaint: I have been threatened with violence, and I am in danger of harm.

Complaint Analysis Results:
Complaint: I have been threatened with violence, and I am in danger of harm.

Applicable IPC Sections:
  - IPC Section 506: Fuzzy Score = 0.9500
  - IPC Section 307: Fuzzy Score = 0.5000

Details:
  IPC 506 - Score: 0.6543, Fuzzy: 0.9500
  IPC 307 - Score: 0.5123, Fuzzy: 0.5000
```

## Conclusion
This system provides an automated way to analyze complaints and match them with relevant IPC sections based on the content. It combines modern NLP techniques with fuzzy logic to handle the ambiguity in textual data and produce meaningful results.

## Future Improvements
- **Extend IPC Knowledge Base**: Add more IPC sections and their associated keywords to expand the coverage.
- **Contextual Understanding**: Integrate more advanced NLP models for better understanding of the complaint context.
- **GUI Integration**: Build a user-friendly graphical interface for easier interaction with the system.

![Screenshot (110)](https://github.com/user-attachments/assets/6b770036-3e40-4a38-b084-7c6879f41798)
