import json

# parse single json object at a time
def parse_file(file):
    for line in file:
        yield json.loads(line)