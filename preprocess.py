from pytesseract import Output
import pytesseract
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class Box:
    def __init__(self, id=1, left=0.0, top=0.0, width=0.0, height=0.0):
        self.id = id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
    
    # Gets the smallest box that contains both these boxes
    @staticmethod
    def union(boxA: "Box", boxB: "Box") -> "Box":
        my_right = boxA.left + boxA.width
        my_bottom = boxA.top + boxA.height
        other_right = boxB.left + boxB.width
        other_bottom = boxB.top + boxB.height

        union_left = min(boxA.left, boxB.left)
        union_top = min(boxA.top, boxB.top)

        return Box(
            id=boxA.id,
            left=union_left,
            top=union_top,
            width=max(my_right, other_right) - union_left,
            height=max(my_bottom, other_bottom) - union_top
        )

    # Returns L1 distance + distance between centers / 25
    # Want both because otherwise huge boxes become a problem
    def distance(self, other: "Box") -> float:
        union = Box.union(self, other)
        union.width -= self.width + other.width
        union.height -= self.height + other.height
        union.width = max(0, union.width)
        union.height = max(0, union.height)
        dist_between_centers = abs(self.left + self.width / 2 - other.left - other.width / 2) + abs(self.top + self.height / 2 - other.top - other.height / 2)
        return union.width + union.height + dist_between_centers / 25.0

    # Returns (distance, other_box)
    def closet_to(self, others: list["Box"]) -> tuple[float, "Box"]:
        best = (float("inf"), Box())
        for other in others:
            dist = self.distance(other)
            if dist < best[0]:
                best = (dist, other)
        return best
    
    def __str__(self):
        return f"{self.id},{self.left},{self.top},{self.width},{self.height}"

    @staticmethod
    def from_str(s: str) -> "Box":
        parts = s.split(",")
        return Box(
            id=int(parts[0]),
            left=float(parts[1]),
            top=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4])
        )

def binarize(img, threshold):
    # TODO: Make threshold adaptive to increase robustness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    img = 255 - img
    return img

def tensor_to_cv2(tens):
    as_numpy = tens.detach().numpy()
    as_numpy = as_numpy.swapaxes(0,1)
    as_numpy = as_numpy.swapaxes(1,2)
    as_numpy = as_numpy * 255
    return as_numpy.astype(np.uint8)

def get_boxes(img, conf_threshold=70, box_merge_dist=25.0):
    results = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Get block_num of confident texts
    confident_blocks = set()
    for i in range(len(results["block_num"])):
        if results["conf"][i] > conf_threshold:
            confident_blocks.add(results["block_num"][i])

    # Perform box merging
    boxes: list[Box] = []
    for i in range(len(results["block_num"])):
        if results["level"][i] != 3 or results["block_num"][i] not in confident_blocks:
            continue
        # Ignore big boxes
        if results["width"][i] > img.shape[0] / 2 or results["height"][i] > img.shape[1] / 2:
            continue
        box = Box(
            results["block_num"][i],
            left=results["left"][i],
            top=results["top"][i],
            width=results["width"][i],
            height=results["height"][i]
        )
        (dist, merge_candidate) = box.closet_to(boxes)
        if dist < box_merge_dist:
            boxes.remove(merge_candidate)
            box = Box.union(merge_candidate, box)
        boxes.append(box)
    
    return boxes

def write_boxes(f_loader, out_dir, conf_threshold=70, should_binarize=True, binarize_threshold=125, save_as_new_imgs=True):
    with open(f"{out_dir}/meta.txt", "w") as fout:
        for ix, (images, labels) in tqdm(enumerate(f_loader), total=len(f_loader)):
            fname = f_loader.dataset.samples[ix][0]
            timestamp = fname.split("f_")[-1].split(".")[0]
            color = tensor_to_cv2(images[0])
            bw =  binarize(color, binarize_threshold) if should_binarize else color
            boxes = get_boxes(bw, conf_threshold=conf_threshold)
            # For each box, get the subimage it bounds and write to file
            # Also write metadata about the box
            temp = cv2.imread(fname)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            for box in boxes:
                sub_color = color[box.top:box.top + box.height, box.left:box.left + box.width]
                sub_bw = bw[box.top:box.top + box.height, box.left:box.left + box.width]
                if save_as_new_imgs:
                    cv2.imwrite(f"{out_dir}/color/{timestamp}_{box.id}.jpg", sub_color)
                    cv2.imwrite(f"{out_dir}/bw/{timestamp}_{box.id}.jpg", sub_bw)   
                fout.write(f"{timestamp},{box.id},{box}\n")
                # draw the box onto images[0]
                cv2.rectangle(temp, (box.left, box.top), (box.left + box.width, box.top + box.height), (0, 255, 0), 2)
            # cv2.imwrite(f"temp.jpg", temp) helpful for debuggin

def pre_process(inp_dir, out_dir, conf_threshold=70, should_binarize=True, binarize_threshold=125, save_as_new_imgs=False):
    f_dataset = datasets.ImageFolder(
        root=inp_dir,
        transform=transforms.ToTensor()
    )
    f_loader = DataLoader(f_dataset, batch_size=1, shuffle=False)
    write_boxes(f_loader, out_dir, conf_threshold, should_binarize, binarize_threshold, save_as_new_imgs)
