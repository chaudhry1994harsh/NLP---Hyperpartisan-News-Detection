from xml.etree import ElementTree
import csv
import pandas as pd
import json


class TrainingDataset:
	def __init__( self, _id, hyperpartsan, labeledBy, url, publishedAt, title, content, bias):
		self._id = _id
		self.hyperpartsan = hyperpartsan
		self.labeledBy = labeledBy
		self.url = url
		self.publishedAt = publishedAt
		self.title = title
		self.content = content
		self.bias = bias


file = '../data/articles-training-bypublisher-20181122/articles-training-bypublisher-20181122.xml'
tree = ElementTree.parse(file)
root = tree.getroot()
tree_T = ElementTree.parse('../data/ground-truth-training-bypublisher-20181122/ground-truth-training-bypublisher-20181122.xml')
root_T = tree_T.getroot()

# print(root)
# print(root_T[0].attrib['hyperpartisan'])
# print(root_T[0].attrib)



def element_to_string(element):
    s = element.text or ""
    for sub_element in element:
        s += ElementTree.tostring(sub_element, encoding='unicode')
    s += element.tail
    return s

i=0

for article in root.iter('article'):    
    if 'id' in article.attrib:
        articleID = article.attrib['id']
    else:
        articleID = ''
    if 'published-at' in article.attrib:
        pub = article.attrib['published-at']
    else:
        pub = ''
    if 'title' in article.attrib:
        articleTitle = article.attrib['title']
    else:
        articleTitle = ''

    contt = element_to_string(article)
    
    hyperpartisan = ''
    labeled_by = ''
    url = ''
    if 'hyperpartisan' in root_T[i].attrib:
        hyperpartisan = root_T[i].attrib['hyperpartisan']
    if 'labeled-by' in root_T[i].attrib:
        labeled_by = root_T[i].attrib['labeled-by']
    if 'url' in root_T[i].attrib:
        url = root_T[i].attrib['url']
    if 'bias' in root_T[i].attrib:
        bias = root_T[i].attrib['bias']

    i += 1


    result = [articleID, pub, articleTitle, contt, hyperpartisan, labeled_by, url, bias]
    # print(result)

    tds = TrainingDataset(articleID, hyperpartisan, bias, labeled_by, url, pub, articleTitle, contt)
    jsonStr = json.dumps(tds.__dict__)

    # print(jsonStr)
    
    with open('articles-training-bypublisher.json', 'a', encoding='utf-8') as f:
        # f.write(jsonStr, f, ensure_ascii=False, indent=4)
        f.write(jsonStr)

    f.close()
    #df = pd.DataFrame(result)
    #df.to_csv('file2.csv', index=False, header=False)

    # with open('data.json', 'a', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, sort_keys = True, indent=4)

# tree_T = ElementTree.parse('data/ground-truth-training-byarticle-20181122/ground-truth-training-byarticle-20181122.xml')
# root_T = tree_T.getroot()

# print(root_T[0].attrib['hyperpartisan'])

# print(root_T[0].attrib)


# tds = TrainingDataset("_id", "hyperpartsan", "labeledBy", "url", "publishedAt", "title", "content")

# print (tds.content)

# jsonStr = json.dumps(tds.__dict__)

# print(jsonStr)


