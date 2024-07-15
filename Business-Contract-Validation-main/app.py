import os
import re
import time
import logging
import pandas as pd
import numpy as np
import torch
import joblib
import pdfplumber
from docx import Document
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util
import faiss
import concurrent.futures
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import uuid
from sklearn.model_selection import train_test_split
from transformers import pipeline
from openvino.runtime import Core
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from openvino.runtime import Core
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

# Initialize components
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(filename='contract_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Utility functions
def log_processing(file_path):
    logging.info(f"Processing file: {file_path}")

def extract_text_from_pdf(file_path):
    logging.info(f"Extracting text from PDF: {file_path}")
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
        logging.info(f"Successfully extracted text from PDF: {file_path}")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {file_path} - {e}")
    return text

def extract_text_from_docx(file_path):
    logging.info(f"Extracting text from DOCX: {file_path}")
    text = ""
    try:
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        logging.info(f"Successfully extracted text from DOCX: {file_path}")
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {file_path} - {e}")
    return text

def preprocess_text_with_spacy(text):
    logging.info(f"Preprocessing text with spaCy")
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    logging.info(f"Successfully preprocessed text with spaCy")
    return " ".join(tokens)

def segment_clauses_advanced(text):
    doc = nlp(text)
    clauses = []
    current_clause = ""
    
    for sent in doc.sents:
        if re.match(r'^\s*(\d+\.|\([a-z]\)|\([0-9]+\)|Section|Article)', sent.text.strip()):
            if current_clause:
                clauses.append(current_clause.strip())
            current_clause = sent.text
        else:
            current_clause += " " + sent.text
    
    if current_clause:
        clauses.append(current_clause.strip())
    
    return clauses

def post_process_clauses(clauses, min_length=50, max_length=1000):
    processed_clauses = []
    current_clause = ""
    
    for clause in clauses:
        clause = re.sub(r'\t.*$', '', clause)
        
        if re.match(r'^\s*(\d+\.|\([a-z]\)|\([0-9]+\)|Section|Article)', clause.strip()):
            if current_clause:
                processed_clauses.append(current_clause.strip())
            current_clause = clause
        elif len(current_clause) + len(clause) < max_length:
            current_clause += " " + clause
        else:
            if current_clause:
                processed_clauses.append(current_clause.strip())
            current_clause = clause
    
    if current_clause:
        processed_clauses.append(current_clause.strip())
    
    return [c for c in processed_clauses if len(c) >= min_length]



def load_standard_templates(templates_dir):
    templates = []
    for filename in os.listdir(templates_dir):
        file_path = os.path.join(templates_dir, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            continue  # Skip unsupported file types
        
        clauses = segment_clauses_advanced(text)
        for clause in clauses:
            templates.append({
                'template_name': filename,
                'clause': clause,
                'category': 'Unknown'  # You may want to add a way to categorize these
            })
    
    return pd.DataFrame(templates)

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

def evaluate_model(classifier, X, y, label_encoder):
    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    
    # Make predictions
    y_pred = classifier.predict(X)
    
    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    # Generate classification report
    class_report = classification_report(y, y_pred, target_names=label_encoder.classes_)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    
    return {
        'cv_scores': cv_scores,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_true': y,
        'y_pred': y_pred
    }

def train_models_from_csv(csv_file_path, models_dir, force_retrain=False):
    model_files = ["vectorizer.pkl", "classifier.pkl", "label_encoder.pkl"]
    models_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)

    if models_exist and not force_retrain:
        print("Models already exist. Loading existing models...")
        vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
        classifier = joblib.load(os.path.join(models_dir, "classifier.pkl"))
        label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        print("Existing models loaded.")
        return vectorizer, classifier, label_encoder

    print("Training new models...")
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Preprocess the clauses
    df['preprocessed_clause'] = df['clause'].apply(preprocess_text_with_spacy)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['preprocessed_clause'], df['category'], test_size=0.2, random_state=42
    )
    
    # Initialize and fit the vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Initialize and fit the label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Initialize and train the classifier
    classifier = SVC(probability=True, random_state=42)
    classifier.fit(X_train_vectorized, y_train_encoded)
    
    # Save the models
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(models_dir, "classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
    
    # Evaluate the model
    evaluation_results = evaluate_model(classifier, X_train_vectorized, y_train_encoded, label_encoder)
    
    # Print or log the results
    print(f"Cross-validation scores: {evaluation_results['cv_scores']}")
    print(f"Average CV score: {np.mean(evaluation_results['cv_scores']):.2f}")
    print(f"Precision: {evaluation_results['precision']:.2f}")
    print(f"Recall: {evaluation_results['recall']:.2f}")
    print(f"F1-score: {evaluation_results['f1']:.2f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])

    # Generate and plot confusion matrix
    cm = evaluation_results['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"\nConfusion Matrix saved as 'confusion_matrix.png' in {models_dir}")

    # Validate the model on test set
    X_test_vectorized = vectorizer.transform(X_test)
    y_test_encoded = label_encoder.transform(y_test)
    accuracy = classifier.score(X_test_vectorized, y_test_encoded)
    
    print(f"\nTest set accuracy: {accuracy:.2f}")

    return vectorizer, classifier, label_encoder

# Define the paths
csv_file_path = r"D:\flks\models\balanced_dataset2.csv"
models_dir = r"D:\flks\models"

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# To force retraining even if models exist:
vectorizer, classifier, label_encoder = train_models_from_csv(csv_file_path, models_dir, force_retrain=False)

# Load the templates
templates_dir = r"D:\flks\lal"
standard_template_df = load_standard_templates(templates_dir) # Implement this function to load your standard template

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from openvino.runtime import Core
import numpy as np
from openvino.tools import mo
import os

def initialize_ner_model():
    # Initialize OpenVINO runtime
    ie = Core()

    # Load the IR model
    model_xml = r"D:\flks\models\ner_model.xml"
    model_bin = r"D:\flks\models\ner_model.bin" # Update this path if necessary
    
    # Read and compile the model
    ov_model = ie.read_model(model_xml, model_bin)
    compiled_model = ie.compile_model(ov_model, "CPU")  # Use "GPU" if available

    # Load the tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load id2label mapping (you might need to save and load this separately)
    id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
                5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

    return compiled_model, tokenizer, id2label

# Initialize the NER model
ner_model, ner_tokenizer, id2label = initialize_ner_model()

# Print the input and output names of the model
print("Model Inputs:", ner_model.inputs)
print("Model Outputs:", ner_model.outputs)

# Assuming 'model' is an instance of the OpenVINO model
inputs = ner_model.inputs
for input_tensor in inputs:
    print(f"Input name: {input_tensor.names}, shape: {input_tensor.shape}")

# Print the input and output names of the model
print("Model Inputs:")
for input_tensor in ner_model.inputs:
    print(f"Input name: {input_tensor.names}, shape: {input_tensor.shape}")

print("Model Outputs:")
for output_tensor in ner_model.outputs:
    print(f"Output name: {output_tensor.names}, shape: {output_tensor.shape}")

def perform_ner(text, model, tokenizer, id2label):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
    
    # Get input_ids
    input_ids = inputs['input_ids']
    
    # Ensure input tensor matches model expected shape
    if input_ids.shape[1] < 512:
        padding_length = 512 - input_ids.shape[1]
        input_ids = np.pad(input_ids, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
    elif input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]

    # Create an inference request
    infer_request = model.create_infer_request()

    # Run inference
    results = infer_request.infer({
        'input_ids': input_ids
    })
    
    # Get the output layer
    output_layer = model.output(0)
    
    # Process the results
    predictions = results[output_layer]
    predictions = np.argmax(predictions, axis=2)
    
    # Convert predictions to entities
    entities = []
    input_ids_list = input_ids[0].tolist()
    for i, prediction in enumerate(predictions[0]):
        if prediction != 0:  # 0 is usually the 'O' (Outside) tag
            word = tokenizer.convert_ids_to_tokens([input_ids_list[i]])[0]
            entity_type = id2label[prediction]
            entities.append((word, entity_type))
    
    return entities

# Example usage
text = "Barack Obama was born in Hawaii."
entities = perform_ner(text, ner_model, ner_tokenizer, id2label)
print(entities)

def hybrid_clause_classification(clause):
    preprocessed_clause = preprocess_text_with_spacy(clause)
    X = vectorizer.transform([preprocessed_clause])
    
    # Ensure the best estimator is used
    best_estimator = classifier.best_estimator_ if hasattr(classifier, 'best_estimator_') else classifier
    
    category_index = best_estimator.predict(X)[0]
    category = label_encoder.inverse_transform([category_index])[0]
    
    # Use predict_proba if available
    if hasattr(best_estimator, 'predict_proba'):
        confidence = np.max(best_estimator.predict_proba(X))
    else:
        confidence = 1.0  # Default confidence if predict_proba is not available
    
    return category, confidence

def find_most_matching_template_optimized(user_clauses_df, standard_template_df, top_k=5):
    logging.info("Finding most matching template using optimized method with GPU and FAISS")
    start_time = time.time()

    template_clauses = standard_template_df['clause'].tolist()
    template_names = standard_template_df['template_name'].tolist()
    template_embeddings = semantic_model.encode(template_clauses, convert_to_tensor=True, batch_size=64)
    template_embeddings = np.array(template_embeddings).astype('float32')

    # Create FAISS index
    index = faiss.IndexFlatIP(template_embeddings.shape[1])
    index.add(template_embeddings)

    best_matches = []

    batch_size = 100
    for i in range(0, len(user_clauses_df), batch_size):
        batch = user_clauses_df.iloc[i:i+batch_size]
        user_embeddings = semantic_model.encode(batch['clause'].tolist(), convert_to_tensor=True, batch_size=64)
        user_embeddings = np.array(user_embeddings).astype('float32')

        # Perform similarity search
        similarities, indices = index.search(user_embeddings, top_k)

        for j, (sim, idx) in enumerate(zip(similarities, indices)):
            clause_matches = [
                {
                    'template_name': template_names[int(i)],
                    'template_clause': template_clauses[int(i)],
                    'similarity': float(s)
                }
                for s, i in zip(sim, idx)
            ]
            best_matches.append({
                'user_clause': batch.iloc[j]['clause'],
                'matches': clause_matches
            })

    # Find the best overall matching template
    template_scores = {}
    for match in best_matches:
        for template_match in match['matches']:
            template_name = template_match['template_name']
            similarity = template_match['similarity']
            if template_name not in template_scores:
                template_scores[template_name] = []
            template_scores[template_name].append(similarity)

    overall_best_match = max(template_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
    best_template_name = overall_best_match[0]
    avg_similarity = sum(overall_best_match[1]) / len(overall_best_match[1])

    logging.info(f"Found most matching template using optimized method with GPU and FAISS in {time.time() - start_time} seconds")
    return best_template_name, avg_similarity, best_matches

def highlight_deviations(user_clauses_df, standard_template_df, clause_matches):
    logging.info("Highlighting deviations")
    deviations = []
    for user_clause, matches in zip(user_clauses_df.itertuples(), clause_matches):
        best_match = max(matches['matches'], key=lambda x: x['similarity'])
        
        similarity = best_match['similarity']
        template_clause = best_match['template_clause']
        
        deviation_percentage = (1 - similarity) * 100

        if deviation_percentage >= 60:
            deviation_type = "High deviation"
        elif 49 <= deviation_percentage < 60:
            deviation_type = "Significant deviation"
        elif 30 <= deviation_percentage < 49:
            deviation_type = "Moderate deviation"
        elif 10 < deviation_percentage < 30:
            deviation_type = "Minor deviation"
        else:
            deviation_type = "Minimal deviation"

        deviations.append({
            'user_clause': user_clause.clause,
            'category': user_clause.predicted_category,
            'similarity': similarity,
            'template_clause': template_clause,
            'deviation_type': deviation_type,
            'deviation_percentage': deviation_percentage,
            'confidence': user_clause.confidence
        })

    return deviations

def prepare_highlighted_contract(user_clauses_df, deviations):
    highlighted_contract = []
    for index, row in user_clauses_df.iterrows():
        user_clause = row['clause']
        deviation = next((d for d in deviations if d['user_clause'] == user_clause), None)
        
        if deviation is not None:
            deviation_percentage = deviation.get('deviation_percentage', None)
            if deviation_percentage is not None:
                if deviation_percentage >= 60:
                    color_class = "high-deviation"
                elif 49 <= deviation_percentage < 60:
                    color_class = "significant-deviation"
                elif 30 <= deviation_percentage < 49:
                    color_class = "moderate-deviation"
                elif 10 < deviation_percentage < 30:
                    color_class = "minor-deviation"
                else:
                    color_class = "minimal-deviation"
                
                highlighted_clause = f'<span class="{color_class}" title="Deviation: {deviation_percentage:.2f}%">{user_clause}</span>'
            else:
                highlighted_clause = user_clause
        else:
            highlighted_clause = user_clause
        
        highlighted_contract.append(highlighted_clause)
    
    return " ".join(highlighted_contract)

def initialize_summarizer():
    # Load BART model and tokenizer for abstractive summarization
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Convert the model to ONNX format
    onnx_model_path = r"D:\flks\models\bart_summarization.onnx"
    dummy_input = tokenizer("This is a test", return_tensors="pt").input_ids

    model.eval()

    torch.onnx.export(model, 
                      (dummy_input,),
                      onnx_model_path,
                      opset_version=14,
                      input_names=['input_ids'],
                      output_names=['output'],
                      dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}},
                      do_constant_folding=True,
                      export_params=True)

    # Initialize OpenVINO runtime and load the model
    ie = Core()
    ov_model = ie.read_model(onnx_model_path)
    compiled_model = ie.compile_model(ov_model, "CPU")

    return compiled_model, tokenizer

summarizer_model, summarizer_tokenizer = initialize_summarizer()

summarizer_model, summarizer_tokenizer = initialize_summarizer()

summarizer_model, summarizer_tokenizer = initialize_summarizer()

print("Summarizer Model Inputs:")
for input_tensor in summarizer_model.inputs:
    print(f"Input name: {input_tensor.get_names()}")

print("Summarizer Model Outputs:")
for output_tensor in summarizer_model.outputs:
    print(f"Output name: {output_tensor.get_names()}")

def abstractive_summarize(text, max_length=70, min_length=50):
    # Tokenize the input text
    inputs = summarizer_tokenizer(text, return_tensors="np", max_length=100, truncation=True)
    
    # Create an inference request
    infer_request = summarizer_model.create_infer_request()
    
    # Convert tokenized inputs to NumPy arrays
    input_ids = inputs['input_ids']
    
    # Prepare inputs for inference
    inputs_dict = {
        'input_ids': input_ids
    }

    # Run inference
    results = infer_request.infer(inputs_dict)
    
    # Get the output
    output = results[summarizer_model.output(0)]

    # Decode the output
    summary_ids = output.argmax(axis=-1)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def extractive_summarize(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:]]
    return ' '.join(ranked_sentences)

def ensemble_summarize(text, max_length=70, min_length=50, num_extractive_sentences=3):
    # Get abstractive summary
    abstractive_summary = abstractive_summarize(text, max_length, min_length)
    
    # Get extractive summary
    extractive_summary = extractive_summarize(text, num_extractive_sentences)
    
    # Combine summaries
    combined_summary = abstractive_summary + " " + extractive_summary
    
    # Tokenize the combined summary
    combined_sentences = nltk.sent_tokenize(combined_summary)
    
    # Remove duplicate sentences
    unique_sentences = list(dict.fromkeys(combined_sentences))
    
    # Join unique sentences
    final_summary = ' '.join(unique_sentences)
    
    return final_summary

def process_contract(file_path, standard_template_df):
    logging.info(f"Starting to process contract: {file_path}")
    
    if file_path.endswith('.pdf'):
        user_text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        user_text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Perform NER on the contract text
    entities = perform_ner(user_text, ner_model, ner_tokenizer, id2label)
    
    # Generate summary
    summary = ensemble_summarize(user_text)
    
    user_clauses = segment_clauses_advanced(user_text)
    user_clauses = post_process_clauses(user_clauses)
    
    classified_clauses = []
    for clause in user_clauses:
        classification, confidence = hybrid_clause_classification(clause)
        classified_clauses.append({'clause': clause, 'predicted_category': classification, 'confidence': confidence})
    
    user_clauses_df = pd.DataFrame(classified_clauses)
    
    best_template_name, avg_similarity, clause_matches = find_most_matching_template_optimized(user_clauses_df, standard_template_df)
    deviations = highlight_deviations(user_clauses_df, standard_template_df, clause_matches)
    highlighted_contract = prepare_highlighted_contract(user_clauses_df, deviations)
    
    logging.info(f"Successfully processed and analyzed contract: {file_path}")
    return {
        'summary': summary,
        'classified_clauses': classified_clauses,
        'best_match': best_template_name,
        'avg_similarity': avg_similarity,
        'deviations': deviations,
        'highlighted_contract': highlighted_contract,
        'clause_matches': clause_matches,
        'entities': entities  # Add the extracted entities to the result
    }

@app.route('/')
def front_page():
    try:
        return render_template('front_page.html')
    except Exception as e:
        logging.error(f"Error rendering front_page.html: {str(e)}")
        return str(e), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            
            try:
                result = process_contract(file_path, standard_template_df)
                return render_template('results.html', 
                                       summary=result['summary'],
                                       deviations=result['deviations'], 
                                       best_match_template=result['best_match'],
                                       highlighted_contract=result['highlighted_contract'],
                                       user_clauses=result['classified_clauses'],
                                       entities=result['entities'])  # Add this line
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                return render_template('error.html', error=str(e)), 500
        else:
            return render_template('error.html', error="Invalid file type"), 400
    
    # If it's a GET request, show the upload form
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form.get('feedback')
    logging.info(f"User feedback received: {feedback}")
    with open('feedback.txt', 'a') as f:
        f.write(f"{feedback}\n")
    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)