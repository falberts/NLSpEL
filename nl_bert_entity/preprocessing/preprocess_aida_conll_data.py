import os
import pickle
import subprocess
from collections import Counter
from typing import Dict

import tqdm

from pipeline_job import PipelineJob


class CreateAIDACONLL(PipelineJob):
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_lang=nl.ttl.bz2.dict",
                "data/indexes/freebase-links_lang=nl.ttl.bz2.dict",
                "data/indexes/page_lang=nl_ids.ttl.bz2.dict",
                "data/indexes/disambiguations_lang=nl.ttl.bz2.dict",
                "data/benchmarks/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv",
            ],
            provides=[
                "data/benchmarks/aida-yago2-dataset/multinerd_dataset.pickle",
                f"data/versions/{opts.data_version_name}/indexes/found_conll_entities.pickle",
                f"data/versions/{opts.data_version_name}/indexes/not_found_conll_entities.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open("data/indexes/redirects_lang=nl.ttl.bz2.dict", "rb") as f:
            redirects_nl = pickle.load(f)

        redirects_nl_values = set(redirects_nl.values())

        with open("data/indexes/freebase-links_lang=nl.ttl.bz2.dict", "rb") as f:
            fb_to_wikiname_dict = pickle.load(f)

        with open("data/indexes/disambiguations_lang=nl.ttl.bz2.dict", "rb") as f:
            disambiguations_dict = pickle.load(f)

        with open("data/indexes/page_lang=nl_ids.ttl.bz2.dict", "rb") as f:
            page_id_to_wikiname_dict = pickle.load(f)

        conll2003_ner_en = self._download(
            url="https://www.clips.uantwerpen.be/conll2003/ner.tgz",
            folder="data/downloads",
        )

        subprocess.check_call(
            [
                "tar",
                "xzf",
                conll2003_ner_en,
                "-C",
                "data/benchmarks/aida-yago2-dataset/",
            ]
        )

        try:
            subprocess.call(
                [
                    "cat data/benchmarks/aida-yago2-dataset/ner/etc/tags.eng"
                    " data/benchmarks/aida-yago2-dataset/ner/etc/tags.eng.testb"
                    " >"
                    " data/benchmarks/aida-yago2-dataset/ner/eng.all"
                ],
            shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)

        with open("data/benchmarks/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv") as f1:
            merged_el = []
            el = f1.readlines()
            last_entity = None
            for el_i, el_line in enumerate(el):
                if len(el_line.strip()) == 0:
                    merged_el.append([""])
                    continue

                if el_line.startswith("-DOCSTART-"):
                    merged_el.append([el_line.strip()])
                    continue

                el_fields = el_line.strip().split("\t")

                # If the line has only one field, treat as non-entity
                if len(el_fields) == 1:
                    bio_ner = "O"
                    rest = []
                else:
                    # The first field is the token, the second is the BIO-NER tag
                    bio_ner = el_fields[1]
                    rest = el_fields[2:]

                merged_el.append([el_fields[0], bio_ner] + rest)

        ##################################################################
        ##################################################################
        ##################################################################
        ##################################################################


        multinerd_dataset = list()
        mentions = list()
        sentence_nr = 0
        tok_nr = 0
        split = "train"

        print(merged_el[:10])

        for line_nr, line_items in enumerate(tqdm.tqdm(merged_el)):

            if (
                len(line_items) > 0
                and line_items[0].startswith("-DOCSTART- testa")
            ):
                split = "valid"
                print(line_items)
                print(split)

            if (
                len(line_items) > 0
                and line_items[0].startswith("-DOCSTART- testb")
            ):
                split = "test"
                print(line_items)
                print(split)

            if len(line_items) == 1 and line_items[0] == '':
                sentence_nr += 1
                tok_nr = 0
            elif len(line_items) > 0 and line_items[0].startswith("-DOCSTART-"):
                sentence_nr = 0
                multinerd_dataset.append(
                    {
                        "tok": " ".join(line_items),
                        "bio-tag": None,
                        "bio": None,
                        "tag": None,
                        "mention": None,
                        "yago_name": None,
                        "wiki_name": None,
                        "wiki_id": None,
                        "fb_id": None,
                        "doc_start": True,
                        "is_nil": None,
                        "sent_nr": sentence_nr,
                        "tok_nr": tok_nr,
                        "split": split,
                    }
                )
            elif len(line_items) == 2:
                tok_nr += 1
                multinerd_dataset.append(
                    {
                        "tok": line_items[0],
                        "bio-tag": "O",
                        "bio": "O",
                        "tag": None,
                        "mention": None,
                        "yago_name": None,
                        "wiki_name": None,
                        "wiki_id": None,
                        "fb_id": None,
                        "doc_start": False,
                        "is_nil": None,
                        "sent_nr": sentence_nr,
                        "tok_nr": tok_nr,
                        "split": split,
                    }
                )
            elif len(line_items) == 4:
                tok_nr += 1
                multinerd_dataset.append(
                    {
                        "tok": line_items[0],
                        "bio-tag": line_items[1],
                        "bio": line_items[1].split("-")[0],
                        "tag": line_items[1].split("-")[1],
                        "mention": line_items[2],
                        "yago_name": line_items[3],
                        "wiki_name": line_items[3],
                        "wiki_id": line_items[3],
                        "fb_id": line_items[3],
                        "doc_start": False,
                        "is_nil": True,
                        "sent_nr": sentence_nr,
                        "tok_nr": tok_nr,
                        "split": split,
                    }
                )
            elif len(line_items) == 6 or len(line_items) == 7:
                tok_nr += 1
                multinerd_dataset.append(
                    {
                        "tok": line_items[0],
                        "bio-tag": line_items[1],
                        "bio": line_items[1].split("-")[0],
                        "tag": line_items[1].split("-")[1],
                        "mention": line_items[2],
                        "yago_name": line_items[3],
                        "wiki_name": Counter({line_items[4].split("/")[-1]: 1}),
                        "wiki_id": line_items[5],
                        "fb_id": line_items[6] if len(line_items) == 7 else None,
                        "doc_start": False,
                        "is_nil": False,
                        "sent_nr": sentence_nr,
                        "tok_nr": tok_nr,
                        "split": split,
                    }
                )

                if multinerd_dataset[-1]["fb_id"] in fb_to_wikiname_dict:
                    key = fb_to_wikiname_dict[multinerd_dataset[-1]["fb_id"]]
                    if key in redirects_nl:
                        key = redirects_nl[key]
                    multinerd_dataset[-1]["wiki_name"][key] += 1

                if multinerd_dataset[-1]["wiki_id"] in page_id_to_wikiname_dict:
                    key = page_id_to_wikiname_dict[multinerd_dataset[-1]["wiki_id"]]
                    if key in redirects_nl:
                        key = redirects_nl[key]
                    multinerd_dataset[-1]["wiki_name"][key] += 1

                for wn in set(
                    map(redirects_nl.get, multinerd_dataset[-1]["wiki_name"].keys())
                ):
                    if wn not in multinerd_dataset[-1]["wiki_name"]:
                        multinerd_dataset[-1]["wiki_name"][wn] += 1

                if multinerd_dataset[-1]["mention"].replace(" ", "_") in redirects_nl:
                    multinerd_dataset[-1]["wiki_name"][
                        redirects_nl[multinerd_dataset[-1]["mention"].replace(" ", "_")]
                    ] += 1

                for wn in set(multinerd_dataset[-1]["wiki_name"].keys()):
                    if wn in disambiguations_dict:
                        multinerd_dataset[-1]["wiki_name"][wn] = 0

                for wn in set(multinerd_dataset[-1]["wiki_name"].keys()):
                    if wn in redirects_nl_values:
                        multinerd_dataset[-1]["wiki_name"][wn] += 5

            else:
                raise Exception("Error {}".format(line_items))

            if (
                len(multinerd_dataset) > 0
                and multinerd_dataset[-1]["bio"] == "B"
                and not multinerd_dataset[-1]["is_nil"]
            ):
                mentions.append(multinerd_dataset[-1])

            # if len(multinerd_dataset) > 0:
            #     print(multinerd_dataset[-1])
            # if line_nr > 1000:
            #     break

        with open("data/benchmarks/aida-yago2-dataset/multinerd_dataset.pickle", "wb") as f:
            pickle.dump(multinerd_dataset, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/entity_counter.pickle",
            "rb",
        ) as f:
            all_entity_counter = pickle.load(f)

        all_found_conll_entities = set()
        all_conll_entities = set()
        all_not_found_conll_entities = set()

        for item in conll_dataset:
            if not item["is_nil"] and item["bio"] == "B":
                name, count = item["wiki_name"].most_common()[0]
                if name in all_entity_counter:
                    all_found_conll_entities.add(name)
                else:
                    all_not_found_conll_entities.add(name)
                all_conll_entities.add(name)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/found_conll_entities.pickle",
            "wb",
        ) as f:
            pickle.dump(all_found_conll_entities, f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/not_found_conll_entities.pickle",
            "wb",
        ) as f:
            pickle.dump(all_not_found_conll_entities, f)

        self.log(f"Found {len(all_found_conll_entities)} and not found {len(all_not_found_conll_entities)} of AIDA-CoNLL entities in the entities dictionary.")

