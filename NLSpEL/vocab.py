"""
NOTE: This code is adapted from the original SpEL repository (https://github.com/shavarani/SpEL).
"""

import pickle
from transformers import AutoTokenizer


class Vocab:
    def __init__(self, args=None):
        self.tag2idx = None
        self.idx2tag = None
        self.OUTSIDE_ID = None
        self.PAD_ID = None
        self.SPECIAL_TOKENS = None
        self.tokenizer = None
        if args is not None:
            self.load(args)

    def load(self, args, popular_entity_to_id_dict=None):

        if popular_entity_to_id_dict is None:
            with open(f"data/versions/{args.data_version_name}/indexes/popular_entity_to_id_dict.pickle", "rb") as f:
                popular_entity_to_id_dict = pickle.load(f)

        if args.uncased:
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", do_lower_case=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", do_lower_case=False)

        self.tag2idx = popular_entity_to_id_dict

        self.OUTSIDE_ID = len(self.tag2idx)
        self.tag2idx["|||O|||"] = self.OUTSIDE_ID

        self.PAD_ID = len(self.tag2idx)
        self.tag2idx["|||PAD|||"] = self.PAD_ID

        self.SPECIAL_TOKENS = [self.OUTSIDE_ID, self.PAD_ID]

        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        self.tokenizer = tokenizer

        args.vocab_size = self.size()

    def size(self):
        return len(self.tag2idx)