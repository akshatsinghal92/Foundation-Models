import json
import pandas as pd

with open("/home/839temp/prometheus-vision/new_train_data.json", "r") as f:
    data = json.load(f)

print(len(data))

# print(len(data))
# df=pd.DataFrame(data)
# print(df['image'].value_counts())
# print(df['image'].unique()[:10])
# import os
# print(len(os.listdir("/home/839temp/prometheus-vision/images")))
# df.to_csv("formatted_responses_feedback.csv", index=False)
# print(data[0]['image'])

# test_images = ['201.png', '217.png', '216.png', '65.png', '66.png', '199.png', '67.png', '277.png', '276.png', '262.png']
# test_data=[]
# train_data=[]
# for i in data:
#     if i['image'] in test_images:
#         # print(i['image'])
#         test_data.append(i)
#     else:
#         train_data.append(i)


# with open("/home/839temp/prometheus-vision/new_test_data.json", "w") as f:
#     json.dump(test_data, f)

# with open("/home/839temp/prometheus-vision/new_train_data.json", "w") as f:
#     json.dump(train_data, f)

# with open("/home/839temp/prometheus-vision/train_data.json", "r") as f:
#     train_data = json.load(f)

# with open("/home/839temp/prometheus-vision/new_test_data.json", "r") as f:
#     test_data = json.load(f)

# for i, row in enumerate(test_data):
#     test_data[i]['question_id']=i

# # print(test_data[0])

# with open("/home/839temp/prometheus-vision/new_test_data.json", "w") as f:
#     json.dump(test_data, f)

# print(len(train_data))
# print(len(test_data))