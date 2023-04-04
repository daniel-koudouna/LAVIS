import torch
import os

from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset

def to_answer(holds):
    if holds:
        return "Yes"
    return "No"


class KittiDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["filename"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["boolean_question"])

        answers = [to_answer(ann["holds"])]
        # TODO this should be configured better
        weights = [0.2]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }
