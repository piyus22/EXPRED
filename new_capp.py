import os
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from Bio.SeqIO.FastaIO import SimpleFastaParser
from sklearn.svm import SVC  # Ensure sklearn's SVC is imported to prevent loading issues

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'fasta', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Load models
try:
    print("Loading model files...")
    with open('SVM_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('SVMmodel_DP.pkl', 'rb') as f:
        model_2 = pickle.load(f)
    print("Model files loaded successfully.")
except Exception as e:
    print("Error loading model files:", e)
    model, model_2 = None, None

@app.route('/')
def home():
    print("Rendering home page.")
    return render_template('home.html')

@app.route('/predictor/')
def predictor():
    print("Rendering predictor page.")
    return render_template('predictor.html')

@app.route('/about/')
def about():
    print("Rendering about page.")
    return render_template('contact.html')

@app.route('/comparison/')
def comparison():
    print("Rendering comparison page.")
    return render_template('comparison.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request.")
    sequence = request.form.get("sequence", "").strip()
    if not sequence or not sequence.startswith(">"):
        print("Invalid sequence format.")
        return render_template('predictor.html', prediction_text='Invalid format or header missing')
    
    sequences = {}
    seq_id, seq_str = "", ""
    for line in sequence.splitlines():
        if line.startswith(">"):
            if seq_id:
                sequences[seq_id] = seq_str
            seq_id, seq_str = line.strip(), ""
        else:
            seq_str += line.strip()
    if seq_id:
        sequences[seq_id] = seq_str
    print("Parsed sequences:", sequences.keys())
    
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    dipeptides = [a + b for a in amino_acids for b in amino_acids]
    
    feature_vectors = []
    for seq in sequences.values():
        aa_features = [seq.count(aa) / len(seq) for aa in amino_acids]
        dp_features = [seq.count(dp) / len(seq) for dp in dipeptides]
        feature_vectors.append((aa_features, dp_features))
    
    predictions = []
    for aa_feat, dp_feat in feature_vectors:
        pred_aa = model.predict([aa_feat]) if model else [1]
        pred_dp = model_2.predict([dp_feat]) if model_2 else [1]
        
        if pred_aa[0] == 0 and pred_dp[0] == 0:
            predictions.append("Is an Expansin protein sequence")
        elif pred_aa[0] == 1 and pred_dp[0] == 1:
            predictions.append("Is NOT an Expansin protein sequence")
        else:
            predictions.append("Could be an Expansin protein sequence")
    
    print("Predictions:", predictions)
    return render_template("show.html", data=predictions, varchar=list(sequences.keys()))

@app.route('/submit', methods=['POST'])
def submit():
    print("Received file upload request.")
    file = request.files.get('file')
    if not file or file.filename == '' or not allowed_file(file.filename):
        print("Invalid file upload.")
        return render_template('predictor.html')
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print("File saved at:", filepath)
    
    predictions, headers = process_file(filepath)
    print("Predictions from file:", predictions)
    return render_template("show.html", data=predictions, varchar=headers)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    print("Processing uploaded file.")
    sequences = {}
    with open(filepath, 'r') as f:
        for title, seq in SimpleFastaParser(f):
            sequences[title] = seq
    print("Parsed sequences from file:", sequences.keys())
    
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    dipeptides = [a + b for a in amino_acids for b in amino_acids]
    
    feature_vectors = []
    for seq in sequences.values():
        aa_features = [seq.count(aa) / len(seq) for aa in amino_acids]
        dp_features = [seq.count(dp) / len(seq) for dp in dipeptides]
        feature_vectors.append((aa_features, dp_features))
    
    predictions = []
    for aa_feat, dp_feat in feature_vectors:
        pred_aa = model.predict([aa_feat]) if model else [1]
        pred_dp = model_2.predict([dp_feat]) if model_2 else [1]
        
        if pred_aa[0] == 0 and pred_dp[0] == 0:
            predictions.append("Is an Expansin protein sequence")
        elif pred_aa[0] == 1 and pred_dp[0] == 1:
            predictions.append("Is NOT an Expansin protein sequence")
        else:
            predictions.append("Could be an Expansin protein sequence")
    
    print("Predictions from file processing:", predictions)
    return predictions, list(sequences.keys())

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
