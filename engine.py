print("Importing libraries... (Step 0 of 4)")
from vid2frames import convert2frames
from preprocess import pre_process
from classify import classify, Classifier, Labeller
from process import process
from typing import Union
import os
import time


# Helper function to print on high-contrast background
def print_info(text:str):
    print(f"\033[1;37;40m{text}\033[0m")

# Helper function to print black text on yellow background
def print_important(text:str):
    print(f"\033[1;30;43m{text}\033[0m")


class Engine:
    """
    The final deliverable engine
    """
    def __init__(self):
        self.input_files = []
        self.identifiers = [] # Internal names for intermediate files
        self.output_dir = ""
        self.classifier: Union[Classifier, None] = None
        self.labeller: Union[Labeller, None] = None
    
    def solicit_files(self):
        """
        Gets the video files and output_dir from the user
        """
        # Get input files
        while True:
            file = input("Enter a video file to process (or nothing to finish):\n")
            if file == "":
                break
            self.input_files.append(file)
        # Set identifiers simply as filenames
        self.identifiers = [text.split("/")[-1].split(".")[0] for text in self.input_files]
        # Check for duplicate identifiers
        assert len(self.identifiers) == len(set(self.identifiers)), "Duplicate identifiers found. Make sure each file has a unique name."
        # Get output directory
        self.output_dir = input("Enter the output directory:\n")
    
    def _prepare_internal(self):
        """
        Once input/output is set, reset folder structure to be ready for intermediate
        and final data
        """
        # Reset the intermediate data
        intermediate_dirs = [f"data/images/{identifier}" for identifier in self.identifiers]
        intermediate_dirs += [f"data/boxes/{identifier}" for identifier in self.identifiers]
        intermediate_dirs += [f"data/boxes/{identifier}/color" for identifier in self.identifiers]
        intermediate_dirs += [f"data/boxes/{identifier}/bw" for identifier in self.identifiers]
        for dir in intermediate_dirs:
            if os.path.exists(dir):
                os.system(f"rm -rf {dir}")
            os.system(f"mkdir {dir}")

        # Reset the output
        if os.path.exists(self.output_dir):
            os.system(f"rm -rf {self.output_dir}")
        os.system(f"mkdir {self.output_dir}")
        for identifier in self.identifiers:
            os.system(f"mkdir {self.output_dir}/{identifier}")
    
    def do_work(self, ix: int) -> float:
        print_info("Converting video to frames... (Step 1 of 4)")
        duration = convert2frames(
            self.input_files[ix],
            f"data/images/{self.identifiers[ix]}",
            fps=1/5 # Capture 1 frame every 5 seconds
        )

        print_info("Pre-processing frames... (Step 2 of 4)")
        pre_process(
            "data/images",
            f"data/boxes/{self.identifiers[ix]}",
            conf_threshold=70,
            should_binarize=True,
            binarize_threshold=100,
            save_as_new_imgs=True
        )

        print_info("Classifying and labelling boxes... (Step 3 of 4)")
        self.classifier, self.labeller = classify(f"data/boxes/{self.identifiers[ix]}")

        print_info("Processing Data... (Step 4 of 4)")
        process(f"data/boxes/{self.identifiers[ix]}", f"{self.output_dir}/{self.identifiers[ix]}", self.classifier, self.labeller)

        return duration
    
    def run(self):
        self.solicit_files()
        self._prepare_internal()

        for ix in range(len(self.input_files)):
            print()
            print_info(f"Processing video {ix+1} of {len(self.input_files)}")
            # Time the work
            start = time.time()
            duration = self.do_work(ix)
            end = time.time()
            print_important(f"Finished processing video {ix+1} of {len(self.input_files)} in {(end-start):.2f} seconds")
            print_important(f"Real-time factor: {duration/(end-start):.2f}")


if __name__ == "__main__":
    engine = Engine()
    engine.run()
