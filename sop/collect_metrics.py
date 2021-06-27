import heapq
from multiprocessing import Pool

import tqdm
import pandas as pd
import numpy as np
from scipy import spatial


from lightning_modules import *


def create_embeddings(model, data_loader):
    embeddings = []
    labels = []

    for batch in tqdm.tqdm(data_loader):
        img, pos_img, label = batch
        output = model(img).numpy()
        embeddings.append(output)
        labels.append(label)

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels


def calculate_metrics(embeddings_path, distance, first_idx, last_idx, progress_bar=True):
    data = pd.read_csv(embeddings_path, header=0, index_col=0)
    labels = data["label"].to_numpy()
    embeddings = data.drop("label", axis=1).to_numpy()
    del data

    top1 = 0
    top5 = 0

    it = zip(embeddings[first_idx:last_idx], labels[first_idx:last_idx])
    if progress_bar:
        it = tqdm.tqdm(it, total=last_idx-first_idx)

    for embedding, label in it:
        distances = heapq.nsmallest(6, zip(embeddings, labels), key=lambda x: distance(x[0], embedding))
        top1 += distances[1][1] == label
        top5 += any([label == result[1] for result in distances[1:]])

    top1 = top1 / (last_idx-first_idx)
    top5 = top5 / (last_idx-first_idx)

    return top1, top5


def collect_metrics(data_file):
    data = pd.read_csv(data_file)
    max_idx = len(data)
    num_workers = 6
    step = int(max_idx / num_workers)
    del data

    import time
    t1 = time.time()

    with Pool(6) as p:

        idxs = [(data_file, spatial.distance.euclidean, idx, idx + step) for idx in range(0, max_idx, step)]
        results = p.starmap(calculate_metrics, idxs)
        top1_e, top5_e = zip(*results)

        idxs = [(data_file, spatial.distance.cosine, idx, idx + step) for idx in range(0, max_idx, step)]
        results = p.starmap(calculate_metrics, idxs)
        top1_c, top5_c = zip(*results)

    print(sum(top1_e) / len(top1_e))
    print(sum(top5_e) / len(top5_e))
    print(sum(top1_c) / len(top1_c))
    print(sum(top5_c) / len(top5_c))
    print("Pool time", time.time() - t1)


if __name__ == "__main__":
    dm = StanfordProductsDataModule("H:\\Dataset\\Stanford_Online_Products\\Ebay_test.txt",
                                             "H:\\Dataset\\Stanford_Online_Products",
                                             batch_size=32)

    stanford_model = StanfordProductsModel.load_from_checkpoint("C:\\Development\\sop\\checkpoints\\resnet_34_best_val-epoch=10-val_loss=0.0646.ckpt")

    data_file = "C:\\Development\\sop\\train_data.csv"

    # with torch.no_grad():
    #     stanford_model.eval()
    #     embeddins, labels = create_embeddings(stanford_model, dm.train_dataloader())
    #     data = pd.DataFrame(embeddins)
    #     data["label"] = labels
    #
    #     data.to_csv(data_file)


    collect_metrics(data_file)



# Top 1 euc 0.6343953253799868
# Top 5 euc 0.7455218948906125

# Test

# Top 1 euc 0.7193333333333334
# Top 5 euc 0.8256666666666667

# Top 1 0.7213333333333334
# Top 5 0.8273333333333334