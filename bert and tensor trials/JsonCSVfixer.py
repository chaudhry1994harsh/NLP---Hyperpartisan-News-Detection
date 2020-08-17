import json
import pandas as pd
from bs4 import BeautifulSoup

#https://www.geeksforgeeks.org/read-json-file-using-python/

#small file
def byArticle():
    st = 'Data Files\\formatted Json\\articles-training-byarticle.json'
    targJSON = 'Data Files\\Clean files\\Json\\articles-training-byarticle.json'
    targCSV = 'Data Files\\Clean files\\CSV\\articles-training-byarticle.csv'
    f = open(st,'r') 
    data = json.load(f) 
    f.close()

    content = []
    hyperpartsan = []
    ids = []
    #https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
    for element in data:
        ids.append(element['_id'])
        soup = BeautifulSoup(element['content'],"html.parser")
        text = soup.get_text()
        text = text.strip()
        content.append(text)
        if element['hyperpartsan'] == 'true':
            hyperpartsan.append(1)
        else:
            hyperpartsan.append(0)

    df = pd.DataFrame({'id':ids,'truth':hyperpartsan,'content':content})

    #https://stackoverflow.com/questions/43413119/forward-slash-in-json-file-from-pandas-dataframe
    json_records = df.to_json(orient ='records') 
    with open(targJSON, 'w') as data_file:
        json.dump(json.loads(json_records), data_file)
    
    #https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/
    df.to_csv(targCSV)

    print ("done")

#for mid file(validation) and large file(training)
def byPublisher(strng):
    if strng == 'validation':
        st = 'Data Files\\formatted Json\\articles-validation-bypublisher.json'
        targJSON = 'Data Files\\Clean files\\Json\\articles-validation-bypublisher.json'
        targCSV = 'Data Files\\Clean files\\CSV\\articles-validation-bypublisher.csv'
    elif strng == 'training':
        st = 'Data Files\\formatted Json\\articles-training-bypublisher.json'
        targJSON = 'Data Files\\Clean files\\Json\\articles-training-bypublisher.json'
        targCSV = 'Data Files\\Clean files\\CSV\\articles-training-bypublisher.csv'

    f = open(st,'r') 
    data = json.load(f) 
    f.close()

    content = []
    hyperpartsan = []
    ids = []
    #https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
    for element in data:
        ids.append(element['_id'])
        soup = BeautifulSoup(element['bias'],"html.parser")
        text = soup.get_text()
        text = text.strip()
        content.append(text)
        if element['hyperpartsan'] == 'true':  
            hyperpartsan.append(1)
        else:
            hyperpartsan.append(0)

    df = pd.DataFrame({'id':ids,'truth':hyperpartsan,'content':content})

    #https://stackoverflow.com/questions/43413119/forward-slash-in-json-file-from-pandas-dataframe
    json_records = df.to_json(orient ='records') 
    with open(targJSON, 'w') as data_file:
        json.dump(json.loads(json_records), data_file)
    
    #https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/
    df.to_csv(targCSV)
    
    print ("done")


byArticle()
byPublisher('validation')
byPublisher('training')