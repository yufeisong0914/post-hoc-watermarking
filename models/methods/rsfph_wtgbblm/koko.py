from string import punctuation


def contains_punctuation(mask_cand):
    return any(p in mask_cand.text for p in punctuation)


class MaskSelector:
    def __init__(
            self, method: str, custom_keywords: list[str],
            keyword_mask: str = "adjacent",
            mask_order_by: str = "dep", exclude_cc: bool = False
    ):
        self.method = method
        self.keyword_mask = keyword_mask
        self.mask_order_by = mask_order_by
        self.num_max_mask = []
        self.custom_keywords = custom_keywords
        self.dep_ordering = [
            'expl', 'cc', 'auxpass', 'agent', 'mark', 'aux', 'prep', 'det', 'prt', 'intj', 'parataxis',
            'predet', 'case', 'csubj', 'acl', 'advcl', 'ROOT', 'preconj', 'ccomp', 'relcl', 'advmod',
            'dative', 'xcomp', 'pcomp', 'nsubj', 'quantmod', 'conj', 'nsubjpass', 'punct', 'poss',
            'dobj', 'nmod', 'attr', 'csubjpass', 'meta', 'pobj', 'amod', 'npadvmod', 'appos', 'acomp',
            'compound', 'oprd', 'nummod'
        ]

        if exclude_cc:
            self.dep_ordering.remove("cc")
            self.dep_ordering = self.dep_ordering[:15] + ["cc"]
        else:
            self.dep_ordering = self.dep_ordering[:15]

        self.pos_ordering = [
            'CCONJ', 'AUX', 'ADP', 'SCONJ', 'DET', 'SPACE', 'INTJ', 'PRON', 'SYM', 'VERB', 'ADV',
            'PUNCT', 'X', 'PART', 'NOUN', 'ADJ', 'PROPN', 'NUM'
        ]

    def _check_mask_candidate(
            self, mask_cand, mask_word, keyword: list = None
    ):
        if (mask_cand not in keyword
                and mask_cand not in mask_word
                and not mask_cand.is_punct
                and not mask_cand.pos_ == "PART"
                and not contains_punctuation(mask_cand)
                and mask_cand.text not in self.custom_keywords
        ):
            return True
        else:
            return False

    def keyword_disconnected(self, sen, keyword):
        max_mask_cnt = len(keyword)
        self.num_max_mask.append(max_mask_cnt)
        mask_word = []
        mask_idx = []
        dep_in_sentence = [t.dep_ for t in sen]
        mask_candidates = []

        for d in self.dep_ordering:
            if d in dep_in_sentence:
                indices = [idx for idx, dep in enumerate(dep_in_sentence) if dep == d]
                for idx in indices:
                    mask_candidates.append(sen[idx])

        if mask_candidates:
            mask_candidates = mask_candidates[:max_mask_cnt]
            for m_cand in mask_candidates:
                if self._check_mask_candidate(m_cand, mask_word, keyword):
                    mask_word.append(m_cand)
                    mask_idx.append(m_cand.i)
                    if len(mask_word) >= max_mask_cnt:
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m - offset for m in mask_idx]

        return mask_idx, mask_word

    def keyword_connected(self, sen, keyword):
        mask_word = []
        mask_idx = []

        if self.keyword_mask == "adjacent":
            offset = sen[0].i
            for k in keyword:
                if k.i - offset < len(sen) - 1:
                    mask_cand = sen[k.i - offset + 1]
                    if self._check_mask_candidate(mask_cand, mask_word, keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)

        elif self.keyword_mask == "child":
            for k in keyword:
                mask_candidates = list(k.children)
                # mask_candidates = mask_candidates[:1]
                for mask_cand in mask_candidates:
                    if self._check_mask_candidate(mask_cand, mask_word, keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break

        elif self.keyword_mask == "child_dep":
            mask_candidates = []
            for k in keyword:
                connected_components = list(k.children)
                dep_in_sentence = [t.dep_ for t in connected_components]
                for d in self.dep_ordering:
                    if d in dep_in_sentence:
                        indices = [idx for idx, dep in enumerate(dep_in_sentence) if dep == d]
                        for idx in indices:
                            mask_candidates.append(connected_components[idx])
                # mask_candidates = mask_candidates
                for mask_cand in mask_candidates:
                    if self._check_mask_candidate(mask_cand, mask_word, keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m - offset for m in mask_idx]

        return mask_idx, mask_word

    def grammar_component(self, text_doc, keyword):
        if self.mask_order_by == "dep":
            text_words_label = [token.dep_ for token in text_doc]
            ordering = self.dep_ordering
        else:
            text_words_label = [token.pos_ for token in text_doc]
            ordering = self.pos_ordering

        mask_candidates_i = []
        for label in ordering:
            for i in range(len(text_words_label)):
                if label == text_words_label[i]:
                    mask_candidates_i.append(i)

        mask_word = []
        mask_idx = []
        if mask_candidates_i:
            if self._check_mask_candidate(m_cand, mask_word, keyword):
                pass

        mask_candidates = []
        for o in ordering:
            if o in text_words_labels:
                indices = [idx for idx, dep in enumerate(text_words_labels) if dep == o]
                for idx in indices:
                    mask_candidates.append(text_doc[idx])

        max_mask_cnt = len(keyword)
        mask_word = []
        mask_idx = []
        if mask_candidates:
            mask_candidates = mask_candidates
            for m_cand in mask_candidates:
                if self._check_mask_candidate(m_cand, mask_word, keyword):
                    mask_word.append(m_cand)
                    mask_idx.append(m_cand.i)
                    if len(mask_word) == max_mask_cnt:
                        break




        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = text_doc[0].i
        mask_idx.sort()
        mask_idx = [m - offset for m in mask_idx]

        return mask_idx, mask_word

    def get_masks(self, text_doc, keyword, ent_keyword):
        """
        Args:
            text:
            keyword: List[Spacy tokens]
            ent_keyword:
        Returns:
        """
        if self.method == "keyword_disconnected":
            return self.keyword_disconnected(text_doc, keyword, ent_keyword)
        elif self.method == "keyword_connected":
            return self.keyword_connected(text_doc, keyword, ent_keyword)
        elif self.method == "grammar":
            return self.grammar_component(text_doc, keyword, ent_keyword)
