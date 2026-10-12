import requests
import json
import sys

url = "http://127.0.0.1:5001/result"

task_id = sys.argv[1]

payload = json.dumps({
  'taskId': task_id
})
headers = {'Content-Type': 'application/json'}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
