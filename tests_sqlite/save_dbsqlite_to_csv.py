import pandas as pd
import sqlite3

comment = "Test"
recommended = "Recommandé"
score = 45

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

# save data in csv file repvet.csv

repvet = pd.read_sql_query('SELECT * from predictions', conn)
repvet.to_csv('./prediction_extractions/repvet.csv', index=False)
