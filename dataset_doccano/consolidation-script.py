from jsonl_split import split_line

from pathlib import Path
from concept_scripts.distant_label import label_json_extended

import subprocess
import json
import os
import re

def run_fmt_files():
    # confirm fmt files
    fmt_script = Path("fmt_files.sh")
    if not fmt_script.exists(): raise FileNotFoundError(f"Shell script not found: {fmt_script}")

    result = subprocess.run(["bash", str(fmt_script)], capture_output=True, text=True)

    if result.returncode == 0:
        print("File formatting completed successfully!")
        print(result.stdout)  # The output logs
    else:
        print("File formatting failed:")
        print(result.stderr)


if __name__=="__main__":
    while True:
        print("------------")
        print("1. Input File & Format")
        print("2. Distant Label")
        print("3. Training and outputting")
        print("4. Run and predict run.slurm")
        print("------------")
        
        option = input("Select menu: ")
        # TODO:
        if int(option) == 1:
            print("===============================================")
            print("Labeling and splitting input .jsonl lectures...")

            # Prompt input file path
            print("Specify input dataset file path (.json / .jsonl / .txt)")
            print("Each JSON line must contain the keys: id, text, Comments, and label: ")
            input_fp = input()
            # input_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
            #                                         'common_set', 'brayden_annotations.jsonl') # EXAMPLE from dataset_doccano/common_set/...
            
            # Append the data file paths to a list to be iterated
            _, ext = os.path.splitext(input_fp)
            files_list = []
            if ext.lower() == '.jsonl': files_list.append()
            elif ext.lower() == '.txt':
                try:
                    with open(input_fp, 'r') as file:
                        for line in file:
                            files_list.append(line.strip())
                except FileNotFoundError:
                    print(f"Error: The file '{input_fp}' was not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            
            # Iterate through each input data file
            # Assume that the json line texts exist under our lecture slide collection (/data_text)
            dataset_fp = os.path.join(os.path.dirname(__file__), '..', 'data_text')
            for f_jsonl in files_list:
                input_fp = f_jsonl
                try:
                    with open(input_fp, 'r') as f:
                        # Extract course information for each line of the input data
                        for line in f:
                            course_subject = ""
                            course_code = 0
                            course_lec_title = ""
                            data_segments = []
                            output_dir = ""

                            current_course = json.loads(line)
                            match_found = False

                            # Match the course information this current line belongs to
                            # Compare text with lectures from /data_text
                            for dirpath, dirnames, filenames in os.walk(dataset_fp):
                                if filenames[0] == 'to_annotate.txt': continue
                                course_folder = dirpath.split('/')[-1].split(' ')[0]

                                for lec_slide in filenames:
                                    with open(os.path.join(dirpath, lec_slide), "r") as lec_txt:
                                        content = lec_txt.read()
                                    
                                    if content != current_course['text']: continue

                                    # Course match is found!
                                    match_found = True
                                    course_subject = course_folder.split('-')[0]
                                    course_code = course_folder.split('-')[1]
                                    course_lec_title = lec_slide

                                    # Prompt data segments for the matched course
                                    print("Found course to split: ", course_subject, course_code, "-", course_lec_title)
                                    seg_input = input("Add a data segment (e.g. dev_set, labeled, test_set, or train_set...): ")
                                    data_segments.append(seg_input)

                                    while True:
                                        inp = input("Add additional data segments, or 'done' to continue: ")
                                        if inp == "done": break
                                        else: data_segments.append(inp)

                                    # Prompt output directory
                                    output_dir = input("Specify target output directory: ")
                                    
                                    # Write line into json object to be split based on data segment input
                                    json_line = {}
                                    json_line["id"] = current_course["id"]
                                    json_line["segment"] = data_segments
                                    json_line["course"] = course_subject.lower() + str(course_code)
                                    json_line["lec"] = course_lec_title
                                    json_line["text"] = current_course["text"]
                                    json_line["label"] = current_course["label"]
                                    json_line["Comments"] = current_course["Comments"]

                                    split_line(json_line, output_dir);

                            # Unable to find the course corresponding to the line, require manual annotation 
                            if not match_found:
                                print("Unable to extract lecture title for id: ", current_course["id"], str(course_code))
                except FileNotFoundError: print("File not found!")

            run_fmt_files()

        elif int(option) == 2:
            print("===============================================")
            print("Distant labeling...")
            
            # Based on diagram
            label_json_extended(
                json_in=Path("dataset_doccano/data/dataset_bio/unlabeled"),
                text_in="",  # only needed if keyword extraction is used
                concept_set_file="gpt_o4_distant_labelling/annotation_results",
                use_keyword_extraction=False,
                out_path=Path("dataset_doccano/data/dataset_bio/distant")
            )
            
        elif int(option) == 3: # training & outputting
            result = None
            crc = input("Running on CRC? (y/n): ")
            if crc == 'y':
                result = subprocess.run(["sbatch", 'p2p-cce-training-script/run.slurm'], capture_output=True, text=True)
            else:
                results = subprocess.run(["sys.executable", 'p2p-cce-training-script/main.py'], capture_output=True, text=True)

            if result.returncode == 0:
                print("Run train slurm success!!")
                print("SLURM response:", result.stdout.strip())
            else: print("Error submitting job:", result.stderr.strip())

        elif int(option) == 4: # run & predict
            result = None
            crc = input("Running on CRC? (y/n): ")

            if crc == 'y':
                result = subprocess.run(["sbatch", 'p2p-cce-training-script/run_predict.slurm'], capture_output=True, text=True)
            else:
                results = subprocess.run(["sys.executable", 'p2p-cce-training-script/main_predict.py'], capture_output=True, text=True)

            if result.returncode == 0:
                print("Run predict slurm success!")
                print("SLURM response:", result.stdout.strip())
            else: print("Error submitting job:", result.stderr.strip())
