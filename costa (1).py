#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import glob
import re
import torch
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from PIL import Image
from tqdm import tqdm
import json
from tabulate import tabulate
import math
from multiprocessing import Pool
import time


# In[3]:


import requests
from bs4 import BeautifulSoup

def get_google_search_hit_count(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    hit_count_element = soup.select_one("#result-stats")
    hit_count = hit_count_element.text if hit_count_element else "0"

    number = re.search(r'[\d.,]+', hit_count).group()
    number = number.replace(',', '')

    return float(number)


# In[ ]:


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def evaluate_and_save_results(all_nearest, y_true):
    top1, top3, top5 = calc_accuracy(all_nearest, y_true)
    return top1, top3, top5
    # top_all[name] = {"top1": top1, "top3": top3, "top5": top5}


def get_similarity(label1,label2):
    q1 = "{} {}".format(label1,label2)
    q2 = "{} {}".format(label2,label1)
    q3 = "{} and {}".format(label1,label2)
    q4 = "{} and {}".format(label2,label1)
    
    # Add a sleep to avoid rate limiting
    time.sleep(2)  # Adjust the sleep duration as needed

    
    c_ij = (get_google_search_hit_count(q1) + get_google_search_hit_count(q2) + get_google_search_hit_count(q3) + get_google_search_hit_count(q4))/4
    c_i = get_google_search_hit_count(label1)
    c_j = get_google_search_hit_count(label2)
    dice = c_ij/(c_i + c_j)
    burst = c_ij**0.5
    return dice#,burst



def score_zero(probab_label_dict, test_labels):
    all_zeros = []
    unseen_scores = {}
    for item in probab_label_dict:
        unseen_scores = {}
        for test_label in test_labels:
            score = sum([similarity_look_up[f"{test_label} - {label}"]*item[label] for label in item])
            unseen_scores[test_label] = score
        unseen_scores = dict(sorted(unseen_scores.items(), key=lambda item: item[1], reverse=True))
        all_zeros.append([list(unseen_scores.keys())[:1],list(unseen_scores.keys())[:3],list(unseen_scores.keys())[:5]])
    return all_zeros 



def calc_accuracy(y_pred, y_true):
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_samples = len(y_true)

    for pred, true in zip(y_pred, y_true):
        nearest_neighbor1, nearest_neighbor3, nearest_neighbor5 = pred
        if true in nearest_neighbor1:
            top1_correct += 1

        if true in nearest_neighbor3:
            top3_correct += 1

        if true in nearest_neighbor5:
            top5_correct += 1

    top1_acc = top1_correct / total_samples
    top3_acc = top3_correct / total_samples
    top5_acc = top5_correct / total_samples

    return top1_acc, top3_acc, top5_acc    


# In[ ]:


train_labels = [line.strip() for line in open('/uufs/chpc.utah.edu/common/home/u1471783/experiments/Animals_with_Attributes2/trainclasses.txt', 'r').readlines()]
unseen_labels = sorted([line.strip() for line in open('/uufs/chpc.utah.edu/common/home/u1471783/experiments/Animals_with_Attributes2/testclasses.txt', 'r').readlines()])
attributes = [re.sub(r'[^a-zA-Z\s]', '', line.split('\t')[1]) for line in [line.strip() for line in open('/uufs/chpc.utah.edu/common/home/u1471783/experiments/Animals_with_Attributes2/predicates.txt', 'r').readlines()]]




binary_relation = np.loadtxt('/uufs/chpc.utah.edu/common/home/u1471783/experiments/Animals_with_Attributes2/predicate-matrix-binary.txt')



label_attr_dict = {}
for i, label in enumerate(train_labels):
    label_attr = dict(zip(attributes,binary_relation[i]))
    filtered_label_attr = {key: value for key, value in label_attr.items() if value == 1.0}
    label_attr_dict[label] =list(filtered_label_attr.keys())

train_labels = sorted(train_labels)


# similarity_look_up = {}
# for i in unseen_labels:
#     for j in train_labels:
#         similarity_look_up[(i,j)] = get_similarity(i,j)

with open('similarity-burst-all.json', 'r') as f:
  similarity_look_up = json.load(f)

print('Done')
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)




model = torch.load("/uufs/chpc.utah.edu/common/home/u1471783/experiments/fine-tuned-models/renet-2-20-epoch.pth")
model = model.to(device)
model.eval()





test_data = {}
path = '/uufs/chpc.utah.edu/common/home/u1471783/experiments/Animals_with_Attributes2/JPEGImages/'
for label in unseen_labels:
    image_files = glob.glob(path+label+'/*')
    for file in image_files:
        test_data[file] = label

print(len(test_data))




class CustomDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = list(data_dict.keys())
        self.targets =list(data_dict.values())
        self.transform = transform

    def __getitem__(self, index):
        # Retrieve an item from the dataset
        image_path = self.data[index]
        label = self.targets[index]

        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)


# Define transformation for your data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Create an instance of your custom dataset
dataset = CustomDataset(test_data, transform=transform)


# In[ ]:


batch_size = 256
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
#         labels = labels.to(device)

        output = model(images).squeeze(0).softmax(0)
        top_probs, top_indices = torch.topk(output, 10)
        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        
        label_prob_dict = [{train_labels[idx]: prob_list[i] for i, idx in enumerate(index_list)} for index_list, prob_list in zip(top_indices, top_probs)]
        unseen_scores = score_zero(label_prob_dict,unseen_labels)
        # print(unseen_scores)
        # nn1,nn3,nn5 = (unseen_scores[:1],unseen_scores[:3],unseen_scores[:5])
        
        y_true.extend(labels)
        y_pred.extend(unseen_scores)


# In[ ]:


table_data = []
top1, top3, top5 = evaluate_and_save_results(y_pred, y_true)
table_data.append(['Costa-Burst', top1, top3, top5])

table_headers = ["Method", "Top 1 Accuracy", "Top 3 Accuracy", "Top 5 Accuracy"]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")
with open('/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/costa-burst.txt', 'w') as f:
    f.write(table)
