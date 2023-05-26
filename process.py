from preprocess import pre_process
from classify import classify
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import pytesseract
import numpy as np


class OutputManager:
    def __init__(self, labels: list[str], output_dir="output", num_remember = 10):
        self.labels = labels
        # Remember the last things we've written to avoid adding to log
        # multiple times from consecutive frames
        self.num_remember = num_remember
        self.memory: dict[str, list[str]] = {lbl:[] for lbl in labels}
        self.output_dir = output_dir
        self.fds = {}
        for label in labels:
            self.fds[label] = open(f"{output_dir}/{label}.json", "w")
            self.fds[label].write('{\n\t"texts": [\n')
    
    def close(self):
        for fd in self.fds.values():
            fd.write("\n\t]\n}\n")
            fd.close()
    
    @staticmethod
    def pretty_timestamp(timestamp):
        seconds = int(int(timestamp) / 1000)
        minutes = int(seconds / 60) % 60
        hours = int(seconds / 3600)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def add_text(self, label, timestamp, text):
        # Remove " and \ from text so JSON doesn't get messed up
        text = text.replace('"', "''").replace("\\", "")
        if text in self.memory[label]:
            return
        if len(self.memory[label]) > 0:
            self.fds[label].write(",\n")
        self.memory[label].append(text)
        while len(self.memory[label]) > self.num_remember:
            self.memory[label].pop(0)
        self.fds[label].write(
            f'\t\t["{OutputManager.pretty_timestamp(timestamp)}", "{text}"]'
        )


class TesseractProcessDataset(Dataset):
    """
    Loads data from pre-processing and classifying in a performant
    format for final text identification using tesseract
    NOTE: Needed because even with optimization, I could not get
    TrOCR to run fast enough when device=CPU. Very similar to dataset above
    """
    def __init__(self, labeller, classifier, color_dir: str):
        self.labeller = labeller
        self.classifier = classifier
        self.color_dir = color_dir
        self.frames_in_order = sorted(classifier.key2ix.keys(), key=lambda x: int(x.split("_")[0]))
    
    def __len__(self):
        return len(self.classifier.key2ix)
    
    def __getitem__(self, ix):
        key = self.frames_in_order[ix]
        timestamp = key.split("_")[0]
        label = self.labeller.ix2label[self.classifier.key2ix[key]]
        filepath = f"{self.color_dir}/{key}.jpg"
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {
            "img": img,
            "label": label,
            "key": key,
            "timestamp": timestamp
        }

def process(input_dir, output_dir, classifier, labeller):
    tesseract_ds = TesseractProcessDataset(labeller, classifier, f"{input_dir}/color")
    tesseract_loader = DataLoader(tesseract_ds, batch_size=1, shuffle=False)
    labels = list(labeller.label2ixs.keys())
    out_man = OutputManager(labels, output_dir=output_dir)

    for batch in tqdm(tesseract_loader):
        img = batch["img"][0]
        img = np.array(img)
        results = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        last_line_num = -1
        text = []
        cur_sentence = ""
        for ix in range(len(results["text"])):
            if last_line_num != -1 and last_line_num != results["line_num"][ix]:
                text.append(cur_sentence)
                cur_sentence = ""
            last_line_num = results["line_num"][ix]
            cur_sentence += results["text"][ix] + " "
        if (len(cur_sentence) > 0):
            text.append(cur_sentence)
        for line in text:
            out_man.add_text(batch["label"][0], batch["timestamp"][0], line)
    
    out_man.close()
