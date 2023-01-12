import sqlite3

comment = "Test2"
recommended = "Non Recommandé"
score = 20

# establish database connection


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn


conn = create_connection('predictions.db')
c = conn.cursor()

# create sqlite database tables

c.execute('''CREATE TABLE IF NOT EXISTS predictions (predictions_id INTEGER PRIMARY KEY AUTOINCREMENT, predictions_review_text TEXT, predictions_recommended VARCHAR(50), predictions_review_score INTEGER)''')


def create_prediction(conn, predictions):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO predictions(predictions_review_text, predictions_recommended, predictions_review_score)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, predictions)
    conn.commit()
    return cur.lastrowid


create_prediction(conn, (comment, recommended, score))
