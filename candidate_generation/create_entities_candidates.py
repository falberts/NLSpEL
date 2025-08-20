#!/usr/bin/env python3
import csv
import re
import requests
import json
import os
import gc


def split_tsv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar="|")

        docs = []
        doc_ents = []
        first_doc = True

        for row in reader:
            if "-DOCSTART-" in row:
                if first_doc:
                    first_doc = False
                    continue
                docs.append(doc_ents)
                doc_ents = []
            elif len(row) > 0:
                if len(row) > 4 and row[1].split('-')[0] == "B":
                    entity = row[2].replace(' ', '').split('(')[0].lower()
                    doc_ents.append(entity)
        if doc_ents:
            docs.append(doc_ents)
        
    return docs


def extract_entities(file_path):
    with open(file_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar="|")
        entities = {}

        for row in reader:
            if len(row) > 4 and row[1].split('-')[0] == "B":
                entity_dictname = row[2].replace(' ', '').split('(')[0].lower()
                if entity_dictname not in entities.keys():
                    entity_name = row[2]
                    entity_link = row[4]
                    entity_fid = row[6]
                    entity_wname = row[3]
                    entity_wid = row[5]
                    candidates = []
                    entities[entity_dictname] = {
                        'name': entity_name,
                        'link': entity_link,
                        'wname': entity_wname,
                        'fid': entity_fid,
                        'wid': entity_wid,
                        'candidates': candidates
                    }
            del row
    return entities

def extract_candidates(file_path, entities):

    def add_candidate(entity_key, candidate, all_candidates):
        """Helper function to add a candidate to all_candidates."""

        if candidate not in all_candidates:
            all_candidates[candidate] = {
                'id': None,
                'inCount': 0,
                'outCount': 0,
                'links': set(),
                'url': f"https://nl.wikipedia.org/wiki/{candidate.replace(' ', '_')}",
                'name': candidate.replace(' ', '_'),
                'normalName': candidate.lower(),
                'normalWikiTitle': candidate.lower().replace(' ', '_'),
                "references": set([entity_key])
            }
        else:
            all_candidates[candidate]['references'].add(entity_key)
            

    entity_names = {key.lower().replace(' ', '') for key in entities.keys()}
    all_candidates = dict()

    with open(file_path, 'r', encoding='utf-8') as infile:
        data = infile.readlines()

        for line in data:
            line = line.strip()
            try:
                link_regex = r"\[\[([^]]*)\]\]"
                name_match = re.search(link_regex, line)
                if name_match:
                    name = name_match.group(1).split("|")[0].strip()
                    alias_1 = name_match.group(1).split("|")[1].split("(")[0].strip() if "|" in name_match.group(1) else None
                    alias_2 = name_match.group(1).split("|")[0].split("(")[0].strip()

                    # Check for name, alias_1, and alias_2 in entity_names
                    for alias in [name, alias_1, alias_2]:
                        if alias:
                            alias_key = alias.lower().replace(' ', '')
                            if alias_key in entity_names:
                                entity = alias_key
                                candidate = re.search(r'\[\[([^]]*)\]\]', line)
                                candidate = candidate.group(1)
                                candidate = candidate.split("|")[0].strip()
                                add_candidate(entity, candidate, all_candidates)
                                break

            except IndexError:
                try:
                    name = line.split("[[")[1].split("]]")[0].strip()
                    alias_key = name.lower().replace(' ', '')
                    if alias_key in entity_names:
                        add_candidate(alias_key, name, all_candidates)
                except IndexError:
                    pass

    return all_candidates

def extract_candidates_counts(dir_path, all_candidates):
    for file_name in os.listdir(dir_path):
        # if file_name == "nlwiki-20250320-pages-articles1.xml-p1p134538":
        #     pass
        # else:
        #     continue
        file_path = os.path.join(dir_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
            pages = re.split(r'<page.*?>', content)

            del content
            gc.collect()

            pages = [page for page in pages if page.strip() != '']

        for page in pages:
            # extract the title and id from each page
            title = re.search(r'<title>(.*?)</title>', page)
            title = title.group(1)if title else None
            id = re.search(r'<id>(.*?)</id>', page)
            id = id.group(1) if id else None

            links = re.findall(r'\[\[([^]]*)\]\]', page)
            links = [link for link in links if not re.search(r'^(Categorie|Gebruiker|Bestand):', link)]
            if title and title in all_candidates.keys():
                if id:
                    all_candidates[title]['id'] = id

                for link in links:
                    target = link.split('|')[0]
                    if target in all_candidates.keys():
                        all_candidates[target]['inCount'] += 1
                        all_candidates[title]['outCount'] += 1
                        all_candidates[title]['links'].add(target)
        del pages
        gc.collect()

        print(f"Processed file: {file_name}")

    return all_candidates

def replace_links(all_candidates):
    for candidate in list(all_candidates.keys()):
        if not all_candidates[candidate]['id']:
            del all_candidates[candidate]
        else:
            for link in list(all_candidates[candidate]['links']):
                if link in all_candidates:
                    link_id = all_candidates[link]['id']
                    all_candidates[candidate]['links'].remove(link)
                    all_candidates[candidate]['links'].add(link_id)
                else:
                    all_candidates[candidate]['links'].remove(link)
    return all_candidates

def add_candidates_to_entities(entities, all_candidates):
    for candidate in all_candidates:
        for ref in all_candidates[candidate]['references']:
            if ref in entities:
                entities[ref]['candidates'].append(all_candidates[candidate])
    return entities


def extract_ids_from_api(entities):
    '''Retrieves qid from wikipedia-api for each candidate in entities. Should only used for testing.'''
    for ent in entities:
        for candidate in entities[ent]['candidates']:
            try:
                response = requests.get(f"https://nl.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={candidate}&format=json")
                data = json.loads(response.text)
                id = list(data['query']['pages'].keys())[0]
                if id == "-1":
                    id = None
                try:
                    qid = data['query']['pages'][id]['pageprops']['wikibase_item']
                except:
                    qid = None
                print(f"{entities[ent]['name']} | {candidate} | {id} | {qid}")
            except:
                pass

def export_data(output_directory, split_nr, split, entities):
    # write to files:
    with open(os.path.join(output_directory, str(split_nr)), 'w', encoding='utf-8') as outfile:
        for ent in split:
            output_ent_str = (
                f"ENTITY\ttext:{entities[ent]['name']}\turl:{entities[ent]['link']}\twname:{entities[ent]['wname']}\tid:{entities[ent]['wid']}\tfreebaseId:{entities[ent]['fid']}")
            outfile.write(output_ent_str + "\n")
            for candidate in entities[ent]['candidates']:
                candidate['popularity'] = int(candidate['inCount']) + int(candidate['outCount'])
            
            top_candidates = sorted(
                entities[ent]['candidates'],
                key=lambda c: c['popularity'],
                reverse=True
            )[:20]

            for candidate in top_candidates:
                output_can_str = (
                    f"CANDIDATE\tid:{candidate['id']}\tname:{candidate['name']}\tinCount:{candidate['inCount']}\toutCount:{candidate['outCount']}\tlinks:{candidate['links']}\turl:{candidate['url']}\tnormalName:{candidate['normalName']}\tnormalwikititle:{candidate['normalWikiTitle']}")
                outfile.write(output_can_str + "\n")


def main():
    entity_path = "MULTINERD-dataset.tsv"
    nlwiki_dir_path = "./nlwiki/"
    pagelinks_path = "pagelinks-counts.txt"
    pagelinks_path = "pagelinks-freq.txt"

    entities = extract_entities(entity_path)
    
    all_candidates = extract_candidates(pagelinks_path, entities)

    # testing purposes only:
    # extract_ids_from_api(entities)

    all_candidates = extract_candidates_counts(nlwiki_dir_path, all_candidates)

    all_candidates = replace_links(all_candidates)

    entities_dict = add_candidates_to_entities(entities, all_candidates)

    # create dictionary if doesnt exist
    output_dir = "./multinerd_candidates/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory created: {output_dir}")
    
    doc_splits = split_tsv_file(entity_path)

    for split_nr, split in enumerate(doc_splits):
        export_data(output_dir, split_nr, split, entities_dict)


if __name__ == "__main__":
    main()

