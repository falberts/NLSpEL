import os
import pickle
import json

def main():
    dirpath = "./data/benchmarks/multinerd-dataset/"
    # print(os.listdir(dirpath))
    print(os.listdir("./data/benchmarks"))
    output_dirpath = "./json_data/"
    filepath = os.path.join(dirpath, "dataset_multinerd_254-20.pickle")

    output_dict = {"train": [], "valid": [], "test": []}

    current_split = 0

    with open(filepath, "rb") as file:
        data = pickle.load(file)
        split_mapping = {0: "train", 1: "valid", 2: "test"}
        for split in data:
            print(f"{len(split)}")
            for line in split:
                output_dict[split_mapping[current_split]].append(line)
            current_split += 1
    
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    output_filepath = os.path.join(output_dirpath, "multinerd.json")
    with open(output_filepath, "w", encoding="utf-8") as output_file:
        json.dump(output_dict, output_file, ensure_ascii=False)
    print(f"Data saved to {output_filepath}")


if __name__ == "__main__":
    main()