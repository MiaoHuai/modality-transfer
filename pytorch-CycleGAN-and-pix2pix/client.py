import requests
import json

url = "http://127.0.0.1:5001/"

payload1 = json.dumps({
  "configs": [
      {
          "configId": 1,
          "filePaths": ["datasets/test/opt2sar/1.png"],
          "type": "Vis",
          "outputPath": "result/opt2ir",
          "modalityAugmentConfig": "Vis-SAR"
          # 模态扩增的可选项，【Vis-SAR】【SAR-Vis】【Vis-Infrared】【Infrared-Vis】
      }#,
      # {
      #     "configId": 2,
      #     "filePaths": ["datasets/test/sar2opt/1.png"],
      #     "type": "SAR",
      #     "outputPath": "result/ir2opt",
      #     # "diversityAugmentTimes": 3,
      #     "modalityAugmentConfig": "SAR-Vis"
      #     # 模态扩增的可选项，【Vis-SAR】【SAR-Vis】【Vis-Infrared】【Infrared-Vis】
      # }
  ]
  
})



headers = {'Content-Type': 'application/json'}

response = requests.request("POST", url, headers=headers, data=payload1)
print(response.text)
