import os
import zipfile
from spel.configuration import get_multinerd_yago_tsv_file_path, get_resources_dir

TRAIN_START_LINE = "-SPLITSTART- train"
TESTA_START_LINE = "-SPLITSTART- testa"
TESTB_START_LINE = "-SPLITSTART- testb"

CANONICAL_REDIRECTS = None


class AnnotationRecord:
    def __init__(self, line):
        """
        Lines with tabs are tokens the are part of a mention:
            - column 1 is the token
            - column 2 is either B (beginning of a mention) or I (continuation of a mention)
            - column 3 is the full mention used to find entity candidates
            - column 4 is the corresponding YAGO2 entity (in YAGO encoding, i.e. unicode characters are backslash encoded and spaces are replaced by underscores, see also the tools on the YAGO2 website), OR --NME--, denoting that there is no matching entity in YAGO2 for this particular mention, or that we are missing the connection between the mention string and the YAGO2 entity.
            - column 5 is the corresponding Wikipedia URL of the entity (added for convenience when evaluating against a Wikipedia based method)
            - column 6 is the corresponding Wikipedia ID of the entity (added for convenience when evaluating against a Wikipedia based method - the ID refers to the dump used for annotation, 2010-08-17)
            - column 7 is the corresponding Freebase mid, if there is one (thanks to Massimiliano Ciaramita from Google Zürich for creating the mapping and making it available to us)
        """
        data_columns = line.split('\t')
        self.token = None
        self.begin_inside_tag = None
        self.full_mention = None
        self.mn_entity = None
        self.wikipedia_url = None
        self.wikipedia_id = None
        self.freebase_mid = None
        self.candidates = None
        if data_columns:
            self.token = data_columns[0]
        if len(data_columns) > 1:
            self.begin_inside_tag = data_columns[1]
        if len(data_columns) > 2:
            self.full_mention = data_columns[2].replace(" - ", "-").replace(" e ", "e ")
        if len(data_columns) > 3:
            self.mn_entity = data_columns[3]
        if len(data_columns) > 4:
            self.wikipedia_url = data_columns[4]
        if len(data_columns) > 5:
            self.wikipedia_id = data_columns[5]
        if len(data_columns) > 6:
            self.freebase_mid = data_columns[6]

    def set_candidates(self, candidate_record):
        self.candidates = candidate_record
        self.candidates.non_considered_word_count -= 1

    def __str__(self):
        res = ""
        t = [self.token, self.begin_inside_tag, self.full_mention, self.mn_entity, self.wikipedia_url,
             self.wikipedia_id, self.freebase_mid]
        for ind, e in enumerate(t):
            if not e:
                continue
            if ind < len(t) - 1:
                res += e + "|"
            else:
                res += e
        if res[-1] == "|":
            res = res[:-1]
        return res


class Document:
    def __init__(self, document_id):
        self.document_id = document_id
        self.annotations = []
        self.current_annotation = []

    def add_annotation(self, line, candidates):

        def normalize_mention_text(s: str):
            if not s:
                return ""
            s = s.replace("_", " ")
            return s.strip().lower()

        if not line.strip() or len(line.split('\t')) < 3:
            self.flush_current_annotation()
        else:
            ar = AnnotationRecord(line)
            for c in candidates:
                if ar.full_mention is not None:
                    if c.non_considered_word_count < 1:
                        continue
                    if normalize_mention_text(c.orig_text) == normalize_mention_text(ar.full_mention):
                        # print(c.orig_text, ar.full_mention)
                        ar.set_candidates(c)
                        break
            self.current_annotation.append(ar)

    def flush_current_annotation(self):
        self.annotations.append(self.current_annotation)
        self.current_annotation = []


class Candidate:
    def __init__(self, candidate_line):
        self.id = ""
        self.in_count = 0
        self.out_count = 0
        self.links = []
        self.url = ""
        self.name = ""
        self.normal_name = ""
        self.normal_wiki_title = ""

        for item in candidate_line.split('\t'):
            if item == 'CANDIDATE' or not item.strip():
                continue
            key, _, value = item.partition(':')
            key = key.strip().lower()
            value = value.strip()

            if key == 'id':
                self.id = value
            elif key == 'incount':
                self.in_count = int(value)
            elif key == 'outcount':
                self.out_count = int(value)
            elif key == 'links':
                value = value.strip("{} ")
                self.links = [v.strip(" '") for v in value.split(',') if v.strip()] if value else []
            elif key == 'url':
                self.url = value
            elif key == 'name':
                self.name = value
            elif key == 'normalname':
                self.normal_name = value
            elif key == "normalwikititle":
                self.normal_wiki_title = value.rstrip(')')
            else:
                pass

    def __str__(self):
        return f"id: {self.id}\twiki_page: {self.url}"


class CandidateRecord:
    def __init__(self, entity_header):
        self.candidates = []
        self.text = ""
        self.url = ""
        self.wname = ""
        self.id = ""
        self.freebase_id = ""

        for item in entity_header.split('\t'):
            if item == 'ENTITY':
                continue
            key, _, value = item.partition(':')
            key = key.strip().lower()
            value = value.strip()

            if key == 'text':
                self.text = value
            elif key == 'url':
                self.url = value
            elif key == 'wname':
                self.wname = value
            elif key == 'id':
                self.id = value
            elif key == 'freebaseid':
                self.freebase_id = value
            else:
                pass

        self.orig_text = self.wname
        self.non_considered_word_count = len(self.orig_text.split('_'))

    def add_candidate(self, candidate_line):
        self.candidates.append(Candidate(candidate_line))

    def __str__(self):
        cnds = '\n\t'.join([str(x) for x in self.candidates])
        return f"original_text: {self.orig_text}\tcandidates:\n\t{cnds}"


def get_candidates(multinerd_candidates_zip, last_document_id):
    candidates_string = multinerd_candidates_zip.read("multinerd_candidates/" + str(last_document_id)).decode("utf-8").split("\n")
    candidates = []
    for c_line in candidates_string:
        if not c_line.strip():
            continue
        if c_line.startswith("ENTITY"):
            candidates.append(CandidateRecord(c_line))
        elif c_line.startswith("CANDIDATE"):
            assert len(candidates)
            candidates[-1].add_candidate(c_line)
        else:
            raise ValueError("This must be unreachable!")
    return candidates

class MULTINERDDataset:
    def __init__(self):
        super(MULTINERDDataset, self).__init__()
        self.dataset = None
        self.data_path = str(get_multinerd_yago_tsv_file_path().absolute())
        assert os.path.exists(self.data_path), f"The passed dataset address: {self.data_path} does not exist"
        self.load_dataset()

    def load_dataset(self):
        multinerd_candidates_zip = zipfile.ZipFile(get_resources_dir() / "data" / "multinerd_candidates.zip", "r")
        annotations = [[], [], []]
        current_document = None
        current_document_candidates = None
        data_split_id = -1
        last_document_id = 0

        with open(self.data_path, "r", encoding="utf-8") as data_file:
            for ind, line in enumerate(data_file):
                line = line.strip()
                if line == TRAIN_START_LINE or line == TESTA_START_LINE or line == TESTB_START_LINE:
                    data_split_id += 1
                    print(f"Switching to split {data_split_id} after processing {last_document_id} documents.")

                elif line.startswith("-DOCSTART-"):
                    if current_document:
                        annotations[data_split_id].append(current_document)
                        last_document_id += 1
                    current_document = Document(last_document_id)
                    current_document_candidates = get_candidates(multinerd_candidates_zip, last_document_id)
                else:
                    current_document.add_annotation(line, current_document_candidates)
            if current_document:
                annotations[data_split_id].append(current_document)

        self.dataset = {"train": annotations[0], "testa": annotations[1], "testb": annotations[2]}
        print(len(self.dataset["train"]), "train documents")
        print(len(self.dataset["testa"]), "testa documents")
        print(len(self.dataset["testb"]), "testb documents")
        multinerd_candidates_zip.close()

