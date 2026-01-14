"""
NOTE: This code is adapted from the original SpEL repository (https://github.com/shavarani/SpEL).

This script uses the EntityEvaluationScores which is a minor add-on to the evaluation script released by Nicola De Cao:
    https://github.com/nicola-decao/efficient-autoregressive-EL
and performs a local evaluation of the SpEL fine-tuned models (in different fine-tuning steps).
"""
import re
from tqdm import tqdm
from collections import defaultdict


from spel.model import SpELAnnotator
from spel.utils import chunk_annotate_and_merge_to_phrase, postprocess_annotations, \
    get_multinerd_set_phrase_splitted_documents, compare_gold_and_predicted_annotation_documents, \
    fix_problematic_cases
from spel.decao_eval import EntityEvaluationScores
from spel.data_loader import dl_sa
from spel.configuration import device

FINAL_ANNOTATION_POSTPROCESSING_ALLOWED = True
BERT_ANNOTATOR_CHECKPOINT = "nlspel-step-2.pt"


class SpELEvaluator(SpELAnnotator):
    def __init__(self):
        super(SpELEvaluator, self).__init__()

    def annotate(self, nif_collection, **kwargs):
        assert len(nif_collection.contexts) == 1
        context = nif_collection.contexts[0]
        ignore_non_multinerd_vocab = kwargs["ignore_non_multinerd_vocab"] if "ignore_non_multinerd_vocab" in kwargs else True
        # kb_prefix can either be a single string applied to any prediction or a dictionary defining one specific prefix
        #  for each possible annotation. This dict object MUST contain one entry with the key='[defalt_prefix]' which
        #   will be used for the cases that a model prediction is not found in the object.
        kb_prefix = kwargs["kb_prefix"] if "kb_prefix" in kwargs else 'http://nl.wikipedia.org/wiki/'
        candidates_manager = kwargs["candidates_manager"] \
            if "candidates_manager" in kwargs and kwargs["candidates_manager"] else None
        phrase_annotations = chunk_annotate_and_merge_to_phrase(
            self, context.mention, k_for_top_k_to_keep=1,
            normalize_for_chinese_characters=True)
        if candidates_manager:
            [candidates_manager.modify_phrase_annotation_using_candidates(p, context.mention)
             for p in phrase_annotations]
        last_step_annotations = [[p.words[0].token_offsets[0][1][0],
                                  p.words[-1].token_offsets[-1][1][-1],
                                  (dl_sa.mentions_itos[p.resolved_annotation], p.subword_annotations)]
                                 for p in phrase_annotations if p.resolved_annotation != 0]
        if FINAL_ANNOTATION_POSTPROCESSING_ALLOWED:
            last_step_annotations = postprocess_annotations(last_step_annotations, context.mention)

        canonical_redirects = self.get_canonical_redirects(ignore_non_multinerd_vocab)

        for l_ann in [(l_ann[0], l_ann[1], (
                canonical_redirects[l_ann[2][0]], l_ann[2][1]) if l_ann[2][0] in canonical_redirects else l_ann[2])
                      for l_ann in last_step_annotations]:
            try:
                kbp = kb_prefix[l_ann[2][0]] if type(kb_prefix) == dict else kb_prefix
            except KeyError:
                kbp = kb_prefix['[defalt_prefix]']
            context.add_phrase(
                beginIndex=l_ann[0],
                endIndex=l_ann[1],
                score=sum([x.item_probability() for x in l_ann[2][1]])/len(l_ann[2][1]),
                annotator='http://sfu.ca/spel/annotator',
                taIdentRef=kbp+l_ann[2][0].replace("\"", "%22"))

    def get_model_raw_logits_inference(self, token_ids, return_hidden_states=False):
        encs = self.lm_module(token_ids.to(self.current_device)).hidden_states
        out = self.out_module.weight
        logits = encs[-1].matmul(out.transpose(0, 1))
        # The following line can provide the functionality to mask out any subset of undesired output entities from
        #  the model predictions on each subword in inference time. You don't need to uncomment it if you are just
        # interested in testing SpEL out.
        # logits = dl_sa.get_all_vocab_mask_for_multinerd().unsqueeze(0).unsqueeze(0).repeat(
        #       logits.size(0), logits.size(1), 1) + logits
        return (logits, encs) if return_hidden_states else logits

    def multinerd_mn_evaluate(self, checkpoint_name, k_for_top_k_to_keep=5, ignore_over_generated=False,
                            ignore_predictions_outside_candidate_list=False):
        self.init_model_from_scratch(device=device)
        self.shrink_classification_head_to_multinerd(device=device)
        if checkpoint_name is None:
            print('Loading the model which is fine-tuned on MULTINERD/CoNLL dataset ...')
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=1)
        else:
            self.load_checkpoint(checkpoint_name, device=device)

        # Counters for per-category mention detection and entity linking
        mention_gold_counts = defaultdict(int)
        mention_pred_counts = defaultdict(int)
        mention_tp_counts = defaultdict(int)

        el_gold_counts = defaultdict(int)
        el_pred_counts = defaultdict(int)
        el_tp_counts = defaultdict(int)

        for dataset_name in ['testa', 'testb']:
            evaluation_results = EntityEvaluationScores(dataset_name)
            gold_documents = get_multinerd_set_phrase_splitted_documents(dataset_name)

            for gold_document in tqdm(gold_documents):
                t_sentence = " ".join([x.word_string for x in gold_document])
                t_sentence = fix_problematic_cases(t_sentence)

                predicted_document = chunk_annotate_and_merge_to_phrase(
                    self, t_sentence, k_for_top_k_to_keep=k_for_top_k_to_keep)
                comparison_results = compare_gold_and_predicted_annotation_documents(
                    gold_document, predicted_document, ignore_over_generated=ignore_over_generated,
                    ignore_predictions_outside_candidate_list=ignore_predictions_outside_candidate_list)

                # helper: find category for a mention string by matching it to gold_document tokens
                def get_category_for_mention(mention_text):
                    def normalize(s):
                        return re.sub(r'\s+', ' ', s.strip().lower())
                    norm = normalize(mention_text)
                    words = [t.word_string for t in gold_document]
                    norms = [normalize(w) for w in words]
                    max_span = min(len(words), 40)
                    for i in range(len(words)):
                        joined = norms[i]
                        if joined == norm:
                            val = getattr(gold_document[i], "begin_inside_tag", None)
                            if val:
                                return val.split('-', 1)[-1] if '-' in val else val
                            return 'UNK'
                        for j in range(i + 1, min(i + max_span, len(words))):
                            joined = ' '.join(norms[i:j + 1])
                            if joined == norm:
                                val = getattr(gold_document[i], "begin_inside_tag", None)
                                if val:
                                    return val.split('-', 1)[-1] if '-' in val else val
                                return 'UNK'
                    return 'UNK'

                # process comparisons: e is (gold_ann, pred_ann)
                for e in comparison_results:
                    gold_ann, pred_ann = e[0], e[1]
                    span_begin = getattr(gold_ann, 'begin_character', None) or getattr(pred_ann, 'begin_character', None)
                    span_end = getattr(gold_ann, 'end_character', None) or getattr(pred_ann, 'end_character', None)
                    mention_text = t_sentence[span_begin:span_end] if (span_begin is not None and span_end is not None) else ''

                    category = 'UNK'
                    
                    if mention_text:
                        category = get_category_for_mention(mention_text)

                    gold_has = bool(getattr(gold_ann, 'resolved_annotation', 0))
                    pred_has = bool(getattr(pred_ann, 'resolved_annotation', 0))

                    if gold_has:
                        mention_gold_counts[category] += 1
                    if pred_has:
                        mention_pred_counts[category] += 1
                    if gold_has and pred_has:
                        mention_tp_counts[category] += 1

                    if gold_has:
                        el_gold_counts[category] += 1
                    if pred_has:
                        el_pred_counts[category] += 1
                    if gold_has and pred_has:
                        gold_ent = dl_sa.mentions_itos[getattr(gold_ann, 'resolved_annotation')]
                        pred_ent = dl_sa.mentions_itos[getattr(pred_ann, 'resolved_annotation')]
                        if gold_ent == pred_ent:
                            el_tp_counts[category] += 1

                g_ed = set((e[1].begin_character, e[1].end_character)
                        for e in comparison_results if e[0].resolved_annotation)
                p_ed = set((e[1].begin_character, e[1].end_character)
                        for e in comparison_results if e[1].resolved_annotation)
                g_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[0].resolved_annotation])
                        for e in comparison_results if e[0].resolved_annotation)
                p_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[1].resolved_annotation])
                        for e in comparison_results if e[1].resolved_annotation)
                if p_el:
                    evaluation_results.record_mention_detection_results(p_ed, g_ed)
                    evaluation_results.record_entity_linking_results(p_el, g_el)

            print(evaluation_results)

            def compute_and_print_stats(title, gold_counts, pred_counts, tp_counts):
                cats = sorted(set(list(gold_counts.keys()) + list(pred_counts.keys()) + list(tp_counts.keys())))
                print(f"\n{title} per-category:")
                macro_prec_sum = macro_rec_sum = 0.0
                precs = []
                recs = []
                tp_sum = sum(tp_counts.values())
                pred_sum = sum(pred_counts.values())
                gold_sum = sum(gold_counts.values())
                for c in cats:
                    tp = tp_counts.get(c, 0)
                    pred = pred_counts.get(c, 0)
                    gold = gold_counts.get(c, 0)
                    prec = tp / pred if pred > 0 else 0.0
                    rec = tp / gold if gold > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    print(f"  {c:10s}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  (TP={tp} PRED={pred} GOLD={gold})")
                    precs.append(prec)
                    recs.append(rec)
                macro_p = sum(precs) / len(precs) if precs else 0.0
                macro_r = sum(recs) / len(recs) if recs else 0.0
                macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)) if (macro_p + macro_r) > 0 else 0.0
                micro_p = tp_sum / pred_sum if pred_sum > 0 else 0.0
                micro_r = tp_sum / gold_sum if gold_sum > 0 else 0.0
                micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
                print(f"\n  Macro P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}")
                print(f"  Micro P={micro_p:.4f} R={micro_r:.4f} F1={micro_f1:.4f}\n")

            compute_and_print_stats("Mention detection", mention_gold_counts, mention_pred_counts, mention_tp_counts)
            compute_and_print_stats("Entity linking", el_gold_counts, el_pred_counts, el_tp_counts)

    def fine_tuned_evaluate(self, checkpoint_name):
        self.init_model_from_scratch(device=device)
        self.shrink_classification_head_to_multinerd(device=device)
        if checkpoint_name is None:
            print('Loading the model which is fine-tuned on MULTINERD/CoNLL dataset ...')
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=1)
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        precision, recall, f1, f05, num_proposed, num_correct, num_gold, subword_eval = self.evaluate(0, 2, 1024, 1.1, False)
        print(f"Subword-level evaluation results on wikipedia validation set: "
              f"precision={precision:.5f}, recall={recall:.5f}, f1={f1:.5f}, f05={f05:.5f}, "
              f"num_proposed={num_proposed}, num_correct={num_correct}, num_gold={num_gold}")
        print(subword_eval)


if __name__ == '__main__':
    b_annotator = SpELEvaluator()
    b_annotator.fine_tuned_evaluate(checkpoint_name=BERT_ANNOTATOR_CHECKPOINT)
    b_annotator.multinerd_mn_evaluate(checkpoint_name=BERT_ANNOTATOR_CHECKPOINT)
