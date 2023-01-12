# Imports dans l'application
from flask import send_file
from create_insert_sqlite_tocsv import create_prediction
from create_insert_sqlite_tocsv import create_connection
from create_insert_sqlite_tocsv import save_db_to_csv
# Imports des librairies python
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template
# from flask_debugtoolbar import DebugToolbarExtension
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
# Téléchargement des dépendances nltk
for dependency in (
    "omw-1.4",
    "stopwords",
    "wordnet",
    "punkt",
    "averaged_perceptron_tagger"
):
    nltk.download(dependency)

app = Flask(__name__)  # creation application
app.debug = True
# toolbar = DebugToolbarExtension(app)
app.static_folder = 'static'

# LOAD MODEL

MODEL_VERSION = 'lstm_model_rus.h5'  # modèle
MODEL_PATH = os.path.join(os.getcwd(), 'models',
                          MODEL_VERSION)  # path vers le modèle
model = load_model(MODEL_PATH)  # chargement du modèle

# LOAD TOKENIZER

TOKENIZER_VERSION = 'tokenizer_rus.pickle'
TOKENIZER_PATH = os.path.join(os.getcwd(), 'models',
                              TOKENIZER_VERSION)  # path vers le tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# PREPROCESS TEXT

stop_words = stopwords.words('english')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def cleaning(data):
    # 1. Tokenize
    text_tokens = word_tokenize(data.replace("'", "").lower())
    # 2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    # 3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 4. Lemmatize
    POS_tagging = pos_tag(tokens_without_sw)
    wordnet_pos_tag = []
    wordnet_pos_tag = [(word, get_wordnet_pos(pos_tag))
                       for (word, pos_tag) in POS_tagging]
    wnl = WordNetLemmatizer()
    lemma = [wnl.lemmatize(word, tag) for word, tag in wordnet_pos_tag]
    return " ".join(lemma)


# PREDICT

def my_predict(text):
    # Tokenize text
    text_pad_sequences = pad_sequences(tokenizer.texts_to_sequences(
        [text]), maxlen=300)
    # Predict
    predict_val = float(model.predict([text_pad_sequences]))
    recommandation = "Recommandé" if predict_val > 0.5 else "Non Recommandé"
    score = int(predict_val*100)
    return score, recommandation


# HOMEPAGE ROUTE

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')


# RECOMMANDATION ROUTE (GET and POST)

@app.route('/recommandation', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        if request.form['customer_feedback']:
            customer_feedback = str(request.form['customer_feedback'])
            clean_comment = cleaning(customer_feedback)

            score, recommandation = my_predict(clean_comment)

            # Sauvegarde dans la bdd sqlite
            # Connexion à la bdd
            conn = create_connection(os.path.join(
                os.getcwd(), 'prediction_extractions', 'predictions.db'))
            c = conn.cursor()

            # Création de la bdd et les tables sqlite
            c.execute('''CREATE TABLE IF NOT EXISTS predictions (predictions_id INTEGER PRIMARY KEY AUTOINCREMENT, predictions_review_text TEXT, predictions_recommended VARCHAR(50), predictions_review_score INTEGER)''')
            create_prediction(conn, (customer_feedback, recommandation, score))
            save_db_to_csv(conn)

            return render_template('recommandation.html', text=customer_feedback, recommandation=recommandation, score=f"Note estimée : {score}/100")

    else:
        return render_template('recommandation.html')

# DOWNLOAD DATABASE ROUTE (GET and POST)


@app.route('/getRepVetCSV', methods=['GET', 'POST'])
def repvet_csv():
    return send_file(os.path.join(
        os.getcwd(), 'prediction_extractions', 'repvet.csv'),
        mimetype='text/csv',
        download_name='repvet.csv',
        as_attachment=True
    )

# from flask_sqlalchemy import SQLAlchemy
# # AUTHENTIFICATION
# # init SQLAlchemy so we can use it later in our models
# db = SQLAlchemy()

# app.config['SECRET_KEY'] = 'secret-key-goes-here'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

# db.init_app(app)

# # blueprint for auth routes in our app
# from .auth import auth as auth_blueprint
# app.register_blueprint(auth_blueprint)

# # blueprint for non-auth parts of app
# from .main import main as main_blueprint
# app.register_blueprint(main_blueprint)


if __name__ == '__main__':  # faire run l'application
    app.run(debug=True, use_debugger=True)
