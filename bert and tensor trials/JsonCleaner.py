#this is for json format where the format is fixed 
#fix json file: https://stackoverflow.com/questions/51919698/cant-parse-json-file-json-decoder-jsondecodeerror-extra-data
import json
from bs4 import BeautifulSoup

#https://www.geeksforgeeks.org/read-json-file-using-python/

#small file
'''st = 'Clean Data Files\\formatted raw\\clnSMALL.json'
targ = 'Clean Data Files\\tester.json'
'''
#mid file
'''st = 'Clean Data Files\\formatted raw\\clnMID.json'
targ = 'Clean Data Files\\clnMID.json'
'''
#large file 
st = 'Clean Data Files\\formatted raw\\clnLARGE.json'
targ = 'Clean Data Files\\clnLARGE.json'


f = open(st,'r') 
data = json.load(f) 
f.close()

#https://stackoverflow.com/questions/36606930/delete-an-element-in-a-json-object
for element in data:
    element.pop('labeledBy',None)
    element.pop('url',None)
    element.pop('publishedAt',None)
    #element.pop('_id',None)
    element.pop('title',None)

#print (data[1])
#https://stackoverflow.com/questions/23306653/python-accessing-nested-json-data
#print (data[1]['content'])

#https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
for element in data:
    soup = BeautifulSoup(element['content'],"html.parser")
    text = soup.get_text()
    text.strip()
    element['content'] = text
    soup = BeautifulSoup(element['hyperpartsan'],"html.parser")
    text = soup.get_text()
    text.strip()
    if text == 'true':  
        element['hyperpartsan'] = 1
    else:
        element['hyperpartsan'] = 0
    

#print (data[1])
#print (data[1]['content'])

'''
soup = BeautifulSoup(data[0]['content'])
text = soup.get_text()
text.strip()
print(text)
data[0]['content'] = text'''


with open(targ, 'w') as data_file:
    data = json.dump(data, data_file)
print ("done")