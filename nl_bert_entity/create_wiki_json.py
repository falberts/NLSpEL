import os
import pickle
import json
from transformers import RobertaTokenizer


def process_split(split, data_dir, id2entity):

    def map_entities(ents):
        if isinstance(ents, int):
            return id2entity.get(ents, "|||O|||")
        elif isinstance(ents, (list, tuple)):
            return [map_entities(e) for e in ents]
        else:
            return "|||O|||"

    loc_file = os.path.join(data_dir, f"{split}.loc")

    os.makedirs("json_data", exist_ok=False)
    output_file = os.path.join("json_data", f"{split}.jsonl")

    with open(loc_file, "r") as f:
        locs = [line.strip().split("\t") for line in f if line.strip()]

    shard_map = {}
    for shard, offset in locs:
        shard_map.setdefault(shard, []).append(int(offset))

    with open(output_file, "w", encoding="utf-8") as out_f:
        for shard, offsets in shard_map.items():
            dat_file = os.path.join(data_dir, f"{shard}.dat")
            with open(dat_file, "rb") as dat_f:
                for offset in offsets:
                    dat_f.seek(offset)
                    token_ids, mention_entity_ids, mention_entity_probs, mention_probs = pickle.load(dat_f)
                    mention_entity_probs = [ents[0] for ents in mention_entity_probs]
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    mentions = [map_entities(ents)[0] for ents in mention_entity_ids]
                    out_f.write(json.dumps({
                        "tokens": tokens,
                        "mentions": mentions,
                        "mention_entity_probs": mention_entity_probs,
                        "mention_probs": mention_probs
                    }, ensure_ascii=False) + "\n")

indexes = "/data/versions/multinerd/indexes/"
with open(indexes+"popular_entity_to_id_dict.pickle", "rb") as f:
    entity_to_id = pickle.load(f)
id2entity = {v: k for k, v in entity_to_id.items()}

tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base", do_lower_case=True)

for split in ["valid", "test", "train"]:
    process_split(split, "/data/versions/multinerd/wiki_training/integerized/nlwiki", id2entity)
    print(f"Processed {split} split. - ({split}.json)")
