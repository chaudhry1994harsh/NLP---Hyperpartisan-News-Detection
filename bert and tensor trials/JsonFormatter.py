#https://www.geeksforgeeks.org/writing-to-file-in-python/
#https://www.geeksforgeeks.org/python-string-replace/



def fixSHITTYformat(fix):
    if fix == 'training-byarticle':
        st = 'Data Files\\SHITTY Json\\articles-training-byarticle.json'
        targ = 'Data Files\\formatted Json\\articles-training-byarticle.json'
    elif fix == 'validation-bypublisher':
        st = 'Data Files\\SHITTY Json\\articles-validation-bypublisher.json'
        targ = 'Data Files\\formatted Json\\articles-validation-bypublisher.json'
    elif fix == 'training-bypublisher':
        st = 'Data Files\\SHITTY Json\\articles-training-bypublisher.json'
        targ = 'Data Files\\formatted Json\\articles-training-bypublisher.json'

    f = open(st, "r")
    data = f.read()
    f.close()

    data1 = data.replace("}{\"_id\"","}, {\"_id\"")
    data1 = '[' + data1 + ']'

    f = open(targ, "w")
    f.write(data1)
    f.close()

    print("done")


fixSHITTYformat('training-byarticle')
fixSHITTYformat('validation-bypublisher')
fixSHITTYformat('training-bypublisher')