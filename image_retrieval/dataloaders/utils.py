import torch


def collate_fn(batch):
    imgs = []
    captions = []
    input_ids = []
    attention_masks = []
    max_len = 0
    max_width = 0

    for data in batch:
        max_len = max(max_len, data[0].shape[1])
        max_width = max(max_width, data[0].shape[2])

    for data in batch:
        imgs.append(data[0])
        captions.append(data[1])
        input_ids.append(data[2])
        attention_masks.append(data[3])

    imgs = torch.stack(imgs, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    return imgs, captions, input_ids, attention_masks
