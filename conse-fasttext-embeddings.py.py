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



def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def evaluate_and_save_results(all_nearest, y_true):
    top1, top3, top5 = calc_accuracy(all_nearest, y_true)
    return top1, top3, top5
    # top_all[name] = {"top1": top1, "top3": top3, "top5": top5}



def get_embeddings(sentences):
    tokenized_sentences = [sentence.lower().split() for sentence in sentences.split()]
    for sentence in tokenized_sentences:
        word_embeddings = [t_model.get_vector(word) for word in sentence if word in vocab]
    sentence_embedding = sum(word_embeddings) / len(word_embeddings) if word_embeddings else np.zeros((300))
    return sentence_embedding



def compute_lookup(all_data, get_embeddings, label_weight=1.0, attr_weight=1.0):
    embd_look_up_lab = {}
    embd_look_up_attr = {}
    embd_look_up_comb = {}
    embd_look_up_wei = {}

    for label in all_data:
        uni_emb = [get_embeddings(attr) for attr in all_data[label]]
        if uni_emb:
            uni_emb = np.mean(uni_emb, axis=0)
        else:
            uni_emb = np.zeros((300))

        embd_look_up_lab[label] = get_embeddings(label)
        embd_look_up_attr[label] = uni_emb
        embd_look_up_comb[label] = get_embeddings(label) + uni_emb
        embd_look_up_wei[label] = label_weight * get_embeddings(label) + attr_weight * uni_emb

    return embd_look_up_lab, embd_look_up_attr, embd_look_up_comb, embd_look_up_wei


def get_nearest_neighbors(uni_emb, test_labels, topn):
    test_embd = [get_embeddings(current_label) for current_label in test_labels]
    test_embd = np.array(test_embd)
    cosine_similarities = cosine_similarity(uni_emb.reshape(1, -1), test_embd)
    top_indices = np.argsort(cosine_similarities, axis=1)[:, -topn:]

    return [test_labels[i] for i in top_indices[0][::-1]]
    # return [result[0] for result in t_model.similar_by_vector(uni_emb, topn=topn)]


def compute_predictions(cands, embd_lookup):
    uni_emb = [cands[current_label] * embd_lookup[current_label] for current_label in cands]
    uni_emb = np.mean(uni_emb,axis=0)
    nearest_neighbor1 = get_nearest_neighbors(uni_emb, unseen_labels, topn=1)
    nearest_neighbor3 = get_nearest_neighbors(uni_emb, unseen_labels, topn=3)
    nearest_neighbor5 = get_nearest_neighbors(uni_emb, unseen_labels, topn=5)
    return [nearest_neighbor1, nearest_neighbor3, nearest_neighbor5]



def nearest_predictions(combined_data, k):
    preds_lab = []
    preds_attr = []
    preds_comb = []
    preds_wei = []

    for cands in combined_data:
        preds_lab.append(compute_predictions(cands, embd_look_up_lab))
        preds_attr.append(compute_predictions(cands, embd_look_up_attr))
        preds_comb.append(compute_predictions(cands, embd_look_up_comb))
        preds_wei.append(compute_predictions(cands, embd_look_up_wei))

    return preds_lab, preds_attr, preds_comb, preds_wei


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



# Load the text embedding model
info = api.info()
t_model = api.load("fasttext-wiki-news-subwords-300")
# t_model = api.load('glove-wiki-gigaword-300')
vocab = list(t_model.key_to_index.keys())



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


embd_look_up_lab, embd_look_up_attr, embd_look_up_comb, embd_look_up_wei = compute_lookup(label_attr_dict, get_embeddings,0.6,0.4)


print('Done')



# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)




model = torch.load("/uufs/chpc.utah.edu/common/home/u1471783/experiments/fine-tuned-models/renet-lab-9-epoch.pth")
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



batch_size = 256
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



all_nearest_lab = []
all_nearest_attr = []
all_nearest_comb = []
all_nearest_wei = []
y_true = []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
#         labels = labels.to(device)

        output = model(images).squeeze(0).softmax(0)
        top_probs, top_indices = torch.topk(output, 10)
        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        
        top_data = [{train_labels[idx]: prob_list[i] for i, idx in enumerate(index_list)} for index_list, prob_list in zip(top_indices, top_probs)]
        
        nearest_preds_lab,nearest_preds_attr,nearest_preds_com,nearest_preds_wei = nearest_predictions(top_data,10)
        
        y_true.extend(labels)
        all_nearest_lab.extend(nearest_preds_lab)
        all_nearest_attr.extend(nearest_preds_attr)
        all_nearest_comb.extend(nearest_preds_com)
        all_nearest_wei.extend(nearest_preds_wei)
    
    

# Save individual JSON files
save_json(all_nearest_lab, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/label.json')
save_json(all_nearest_attr, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/Attributes.json')
save_json(all_nearest_comb, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/Sum.json')
save_json(all_nearest_wei, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/Weighted-Sum.json')
save_json(y_true, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/y_true.json')

# Evaluate and save comparison results
result_files = [(all_nearest_lab, 'label'),
                (all_nearest_attr, 'Attribute'),
                (all_nearest_comb, 'Sum'),
                (all_nearest_wei, 'Weighted-Sum')]

table_data = []
for nearest_data, name in result_files:
    top1, top3, top5 = evaluate_and_save_results(nearest_data, y_true)
    table_data.append([name, top1, top3, top5])

table_headers = ["Method", "Top 1 Accuracy", "Top 3 Accuracy", "Top 5 Accuracy"]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")

# Save the table to a file
with open('/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/comparison.txt', 'w') as f:
    f.write(table)

# save_json(top_all, '/uufs/chpc.utah.edu/common/home/u1471783/experiments/experiments-git/perf/comparison.json')
        
        
