import torch
import time


def collate_fn(batch):
    start = time.time()
    imgs = []
    captions = []
    input_ids = []
    attention_masks = []

    for data in batch:
        imgs.append(data[0])
        captions.append(data[1])
        input_ids.append(data[2])
        attention_masks.append(data[3])

    imgs = torch.stack(imgs, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    print(time.time() - start)
    return imgs, captions, input_ids, attention_masks
