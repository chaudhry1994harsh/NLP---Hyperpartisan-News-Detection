#https://www.geeksforgeeks.org/writing-to-file-in-python/
#https://www.geeksforgeeks.org/python-string-replace/


#small file
'''st = 'Data Files\\articles-training-byarticle.json'
targ = 'Clean Data Files\\formatted raw\\clnSMALL.json'
'''
#mid file
'''st = 'Data Files\\articles-validation-bypublisher.json'
targ = 'Clean Data Files\\formatted raw\\clnMID.json'
'''
#large file
'''st = 'Data Files\\articles-training-bypublisher.json'
targ = 'Clean Data Files\\formatted raw\\clnLARGE.json'
'''

f = open(st, "r")
data = f.read()
f.close()

data1 = data.replace("}{\"_id\"","}, {\"_id\"")
data1 = '[' + data1 + ']'

f = open(targ, "w")
f.write(data1)
f.close()

print("done")