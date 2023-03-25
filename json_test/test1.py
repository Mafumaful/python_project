# read the json file
import json

with open('test.json') as f:
    data = json.load(f)

# print the data
print(data)
# print the type of data
print(type(data))
# print the data of the key 'name'
print(data['name'])
# print the address of the key 'address'
print(data['address'])
# print the address of the key 'address' and the key 'street'
print(data['address']['street'])
# print the interest of the key 'interest'
print(data['interests'][0])
