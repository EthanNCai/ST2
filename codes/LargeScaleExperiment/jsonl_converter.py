import json

with open('DJI-10.json', 'r') as file:
    data = json.load(file)

output = {'name': data['name'], 'close': []}
for close, date in zip(data['close'], data['date']):
    output['close'].append({'close': close, 'date': date})

with open("DJI-10-F.json", "w") as file:
    json.dump(output, file)
