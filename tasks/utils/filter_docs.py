import sys
import ijson
import json

klasses = set()
with open(sys.argv[2], 'r') as nameFile:
    names = nameFile.readlines()
    for name in names:
        klasses.add(name.strip())

output = []
with open(sys.argv[1], 'r') as data:
    jsonCollect = ijson.items(data, 'item')
    for jsonObject in jsonCollect:
        if 'class_docstring' not in jsonObject:
            continue
        className = jsonObject['klass']
        if className not in klasses:
            continue
        output.append(jsonObject)

            
with open(sys.argv[3], 'w') as filtered:
    filtered.write(json.dumps(output, sort_keys=True, indent=4))


print(str(klasses))

