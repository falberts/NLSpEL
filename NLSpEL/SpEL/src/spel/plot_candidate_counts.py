import zipfile
import matplotlib.pyplot as plt


def extract_candidate_counts(zipfilename):

    current_entity = None
    current_candidate_count = None
    entity_candidate_counts = {}

    with zipfile.ZipFile(zipfilename, 'r') as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                for line in f.readlines():
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith("ENTITY"):

                        if current_entity:
                            entity_candidate_counts[current_entity] = current_candidate_count

                        parts = line.split('\t')
                        for part in parts:
                            if part.startswith("wname:"):
                                current_entity = part[len("wname:"):].strip()
                                current_candidate_count = 0
                                break

                    elif line.startswith("CANDIDATE"):
                        current_candidate_count += 1

    if current_entity:
        entity_candidate_counts[current_entity] = current_candidate_count

    entity_candidate_counts = sorted(entity_candidate_counts.items(), key=lambda item: item[1], reverse=True)

    return entity_candidate_counts

def plot_candidate_counts(candidate_counts):
    entities, counts = zip(*candidate_counts)

    # font size
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Entities')
    plt.ylabel('Number of Candidates')
    plt.title('Candidate Counts per Entity')

    n = len(entities)
    if n == 0:
        plt.xticks([])
    elif n == 1:
        plt.xticks([0], ['0'], rotation=0, ha='center')
        plt.xlim(-0.5, 0.5)
    else:
        num_x_ticks = 5
        if n <= num_x_ticks:
            ticks = list(range(n))
        else:
            step = (n - 1) / float(num_x_ticks - 1)
            # ticks = [int(round(i * step)) for i in range(num_x_ticks)]
            ticks = [0, 10000, 20000, 30000, 40000, n - 1]
            ticks = sorted(set(ticks))
            labels = [str(t) for t in ticks]

        if labels:
            labels[-1] = str(n)
        plt.xticks(ticks, labels, rotation=0, ha='center')
       
        plt.xlim(-0.5, n - 0.5)
        plt.yticks(range(0, max(counts) + 1, max(1, max(counts) // 10)))
 
    # save plot
    plt.savefig('candidate_counts_plot.png')

def entity_candidate_count_per_category(candidate_counts, multinerd_filename):
    category_counts = {}
    candidate_dict = dict(candidate_counts)

    with open(multinerd_filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            if parts[1].startswith("B-"):
                entity = parts[3]
                category = parts[1]

                if entity not in candidate_dict:
                    continue

                if category not in category_counts:
                    category_counts[category] = {'total_count': 0, 'entity_count': 0}                

                category_counts[category]['total_count'] += candidate_dict[entity]
                category_counts[category]['entity_count'] += 1

    avg_category_counts = {cat: data['total_count'] / data['entity_count'] for cat, data in category_counts.items()}

    return avg_category_counts

def main():
    zipfilename = "../../resources/data/multinerd_candidates.zip"

    multinerd_filename = "../../resources/data/MULTINERD-dataset.tsv"

    candidate_counts = extract_candidate_counts(zipfilename)
    plot_candidate_counts(candidate_counts)

    print(f"Total number of entities: {len(candidate_counts)}")

    not_unique = sum(1 for _, count in candidate_counts if count > 1)
    print(f"Non-unique entities: {not_unique}")

    avg_category_counts = entity_candidate_count_per_category(candidate_counts, multinerd_filename)
    avg_category_counts = dict(sorted(avg_category_counts.items(), key=lambda item: item[1], reverse=True))
    
    for category, avg_count in avg_category_counts.items():
        print(f"{category} {avg_count:.2f}")


if __name__ == '__main__':
    main()
