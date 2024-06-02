from flask import Flask, request
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import predictions as pred
from pymongo import MongoClient
from bson import objectid
import json

client = MongoClient("mongodb+srv://ranjeetxsingh:6GDgrKNYOlfnUzSt@cluster1.jups0mh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")

card_data = {}

db = client["BUSINESS_CARDS_DATA"]

cards_collection = db["cards"]

app = Flask(__name__)
app.secret_key = 'card_scanner_app'

docscan = utils.DocumentScan()

@app.route('/', methods=['GET', 'POST'])
def scandoc():
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print('Image saved in = ', upload_image_path)
        # predict the coordinates of the document
        four_points, size = docscan.document_scanner(upload_image_path)
        print(four_points, size)
        if four_points is None:
            message = 'UNABLE TO LOCARE THE COORDINATES OF DOCUMENT: points displayed are random'
            points = [
                {'x':10, 'y':10},
                {'x':120, 'y':10},
                {'x':120, 'y':120},
                {'x':10, 'y':120}
            ]
            
            return render_template('scanner.html', 
                                    points=points, 
                                    fileupload=True,
                                    message=message)
        
        else:
            points = utils.array_to_json_format(four_points)
            message = 'Located the Coordinates of Document using openCV'
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   message=message)
    
    return render_template('scanner.html')


@app.route('/transform', methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        # utils.save_upload_image(magic_color, 'magic_color.jpg')
        filename = 'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR, filename=filename)
        cv2.imwrite(magic_image_path, magic_color)
        
        return 'success'
    except:
        return 'fail'


@app.route('/prediction')
def prediction():
    global card_data
    # load the wrap image
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, 'magic_color.jpg')
    image = cv2.imread(wrap_image_filepath)
    image_bb, results = pred.getPredictions(image)
    
    bb_filename = settings.join_path(settings.MEDIA_DIR, "bounding_box.jpg")
    cv2.imwrite(bb_filename, image_bb)
    card_data = results
    return render_template('predictions.html', results=results)
    

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/addCard', methods=['POST'])
def addCard():
    cards_collection.insert_one(
        {"NAME": card_data["NAME"],
         "ORG": card_data["ORG"],
         "DES": card_data["DES"],
         "PHONE": card_data["PHONE"],
         "EMAIL": card_data["EMAIL"],
         "WEB": card_data["WEB"]
         }
    )
    return render_template('added.html', results=card_data)


@app.route('/allCards')
def allCards():
    allData = cards_collection.find()
    return render_template('allCards.html', allCardsData=allData)



@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        organisation = request.form['org']
        data = cards_collection.find({"ORG": organisation})
        results = list(data)
        if not results:
            return render_template('searchFail.html', data=data)
        else:
            return render_template('searchResult.html', data=results)
    else:
        return render_template('search.html')


if __name__ == "__main__":
    app.run(debug=True)


