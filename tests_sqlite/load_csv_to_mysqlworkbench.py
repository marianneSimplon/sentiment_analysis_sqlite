import mysql.connector
import sys
import pandas as pd


def connect():
    try:
        cnx = mysql.connector.connect(host="localhost", user="root", password="root",
                                      auth_plugin='mysql_native_password', database="sentiment_analysis_bdd")
    except mysql.connector.Error as err:
        cnx = False
        print(err)
        sys.exit(1)
    finally:
        return cnx


conn = connect()
cursor = conn.cursor()

new_predictions = pd.read_csv("./prediction_extractions/repvet.csv")


def get_prediction_data(new_predictions):
    predictions_data = []
    for i in range(len(new_predictions)):
        predictions_id = new_predictions.iloc[i]['predictions_id']
        predictions_recommended = new_predictions.iloc[i]['predictions_recommended']
        predictions_review_text = new_predictions.iloc[i]['predictions_review_text']
        predictions_review_score = new_predictions.iloc[i]['predictions_review_score']
        predictions_data.append((int(predictions_id), predictions_review_text,
                                predictions_recommended, int(predictions_review_score)))
    return predictions_data


print(get_prediction_data(new_predictions))

sql = '''INSERT IGNORE INTO predictions (predictions_id, predictions_review_text, predictions_recommended, predictions_review_score) VALUES (%s, %s,%s, %s)'''
cursor.executemany(sql, get_prediction_data(new_predictions))
conn.commit()
cursor.close()
