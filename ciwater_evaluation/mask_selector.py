from string import punctuation


def _contains_punctuation(token):
    """
    检查给定的词元素是否包含标点符号
    Args:
        token: 词元素
    Returns: true/false
    """
    return any(p in token.text for p in punctuation)


class MaskSelector:
    def __init__(self, custom_keywords: list[str], method: str, mask_order_by: str, exclude_cc=False):
        self.num_max_mask = []
        self.custom_keywords = custom_keywords
        # 句法依存关系标签（Syntactic Dependency Relation Labels），用于描述单词之间在句子中的语法关系
        self.dep_ordering = [
            'expl',  # 指代性成分，用于表示代词或指示代词等指代性成分。
            'cc',  # 并列连词，用于连接并列的词语或短语。
            'auxpass',  # 被动助动词，用于表示被动语态中的助动词。
            'agent',  # 主动语态中的施事者，表示执行动作的实体。
            'mark',  # 标记词，用于引导从句或标记特定的语义关系。
            'aux',  # 助动词，用于构成各种时态、情态和被动语态等。
            'prep',  # 介词，表示名词或代词与其他词语之间的关系。
            'det',  # 限定词，用于修饰名词的词语。
            'prt',  # 小品词，表示与动词形成短语的副词或介词。
            'intj',  # 感叹词，表示强烈的情感或呼喊。
            'parataxis',  # 平行结构，表示并列或平行的结构。
            'predet',  # 前位限定词，表示位于限定词之前的另一个限定词。
            'case',  # 格标记，表示名词或代词在句中的语法格。
            'csubj',  # 从属主语，表示作为从句主语的词语。
            'acl',  # 关系从句修饰词，表示修饰名词的关系从句。
            'advcl',  # 状语从句，表示修饰动词或形容词的状语从句。
            'ROOT',  # 根节点，表示整个句子的根节点。
            'preconj',  # 前连词，用于连接与后文有关的独立成分。
            'ccomp',  # 补语从句，表示作为宾语补足语的从句。
            'relcl',  # 关系从句，表示修饰名词的关系从句。
            'advmod',  # 状语，表示修饰副词、形容词、动词或整个句子的副词短语。
            'dative',  # 间接宾语，表示动作的接收者或受益者。
            'xcomp',  # 扩展补足语，表示作为动词补足语的从句。
            'pcomp',  # 介词补语，表示作为介词补足语的从句。
            'nsubj',  # 主语，表示执行动作或具有某种状态的实体。
            'quantmod',  # 数量修饰语，表示修饰名词的数量词或数词。
            'conj',  # 连接词，表示连接并列的词语或短语。
            'nsubjpass',  # 被动语态中的主语，表示被动语态中执行动作的实体。
            'punct',  # 标点符号，表示句子中的标点符号。
            'poss',  # 所有格，表示名词的所有者。
            'dobj',  # 直接宾语，表示动作的接收者或影响者。
            'nmod',  # 名词修饰语，表示修饰名词的修饰语。
            'attr',  # 属性，表示名词短语中的属性。
            'csubjpass',  # 被动语态中的从句主语，表示被动语态中执行动作的实体。
            'meta',  # 元信息标签，用于表示附加信息或元数据。
            'pobj',  # 介词宾语，表示介词所引导的宾语。
            'amod',  # 形容词修饰语，表示修饰名词的形容词。
            'npadvmod',  # 名词短语修饰语，表示修饰副词、形容词、动词或整个句子的名词短语。
            'appos',  # 同位语，表示解释或重述名词短语的成分。
            'acomp',  # 形容词补语，表示作为形容词补足语的形容词。
            'compound',  # 复合词，表示由多个词组成的复合词。
            'oprd',  # 宾语补足语，表示作为及物动词宾语补足语的成分。
            'nummod',  # 数词修饰语，表示修饰名词的数词。
        ]

        if exclude_cc:
            self.dep_ordering.remove("cc")
            self.dep_ordering = self.dep_ordering[:39] + ["cc"]  # list类型是可以相加的哦！
        else:
            self.dep_ordering = self.dep_ordering[:40]

        self.pos_ordering = [
            'CCONJ',  # 并列连词，表示连接并列的词语或短语。
            'AUX',  # 助动词，用于构成各种时态、情态和被动语态等。
            'ADP',  # 介词，表示名词或代词与其他词语之间的关系。
            'SCONJ',  # 从属连词，引导从句或标记特定的语义关系。
            'DET',  # 限定词，用于修饰名词的词语。
            'SPACE',  # 空格符号，在句法分析中表示空格。
            'INTJ',  # 感叹词，表示强烈的情感或呼喊。
            'PRON',  # 代词，用于代替名词或名词短语。
            'SYM',  # 符号，表示各种符号或特殊字符。
            'VERB',  # 动词，表示动作、状态或行为。
            'ADV',  # 副词，表示修饰动词、形容词或其他副词的词语。
            'PUNCT',  # 标点符号，在句法分析中表示标点符号。
            'X',  # 其他，表示不属于已知词类的其他词语。
            'PART',  # 小品词，表示与动词形成短语的副词或介词。
            'NOUN',  # 名词，表示人、事物、地点或抽象概念。
            'ADJ',  # 形容词，表示描述名词的性质或特征。
            'PROPN',  # 专有名词，表示特定的人、地点、组织或事物的名称。
            'NUM',  # 数词，表示数量或顺序。
        ]
        self.method = method
        if mask_order_by not in ['dep', 'pos']:
            mask_order_by = 'pos'
        self.mask_order_by = mask_order_by

    def return_mask(self, doc_text, all_entity_keywords: list, all_yake_keywords: list):
        if self.method == "keyword_disconnected":
            return self.keyword_disconnected(doc_text, all_entity_keywords, all_yake_keywords)
        elif self.method == "keyword_connected":
            return self.keyword_connected(
                doc_text, all_entity_keywords, all_yake_keywords, type=self.keyword_mask
            )
        elif self.method == "grammar":
            return self.grammar_component(doc_text, all_entity_keywords, all_yake_keywords)

    def keyword_disconnected(self, sen, keyword, ent_keyword):
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
                if self._check_mask_candidate(m_cand, mask_word, keyword, ent_keyword):
                    mask_word.append(m_cand)
                    mask_idx.append(m_cand.i)
                    if len(mask_word) >= max_mask_cnt:
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m - offset for m in mask_idx]

        return mask_idx, mask_word

    def keyword_connected(self, sen, keyword, ent_keyword, type="adjacent"):
        mask_word = []
        mask_idx = []

        if type == "adjacent":
            offset = sen[0].i
            for k in keyword:
                if k.i - offset < len(sen) - 1:
                    mask_cand = sen[k.i - offset + 1]
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
        elif type == "child":
            for k in keyword:
                mask_candidates = list(k.children)
                # mask_candidates = mask_candidates[:1]
                for mask_cand in mask_candidates:
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break
        elif type == "child_dep":
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
                    if self._check_mask_candidate(mask_cand, mask_word, keyword, ent_keyword):
                        mask_word.append(mask_cand)
                        mask_idx.append(mask_cand.i)
                        break

        mask_word = [x[1] for x in sorted(zip(mask_idx, mask_word), key=lambda x: x[0])]
        offset = sen[0].i
        mask_idx.sort()
        mask_idx = [m - offset for m in mask_idx]

        return mask_idx, mask_word

    def grammar_component(self, doc_text, all_entity_keywords: list, all_yake_keywords: list):
        if self.mask_order_by == "dep":
            text_deps = [token.dep_ for token in doc_text]
            ordering = self.dep_ordering
        else:
            text_deps = [token.pos_ for token in doc_text]
            ordering = self.pos_ordering

        # 把doc_sentence中的token按照ordering的顺序筛选出来
        mask_candidates = []  # list[spacy.Token]
        for o in ordering:
            for i in range(len(text_deps)):
                if text_deps[i] == o:
                    mask_candidates.append(doc_text[i])
        # print(mask_candidates)

        mask_words_index, mask_words = [], []  # mask_words的类型是spaCy的Token
        if mask_candidates:
            for candidate in mask_candidates:
                if self._check_mask_candidate(candidate, mask_words, all_entity_keywords, all_yake_keywords):
                    mask_words.append(candidate)
                    mask_words_index.append(candidate.i)
        #             if len(mask_words) == len(all_entity_keywords) + len(all_yake_keywords):
        #                 break
        #
        # mask_words = [x[1] for x in sorted(zip(mask_words_index, mask_words), key=lambda x: x[0])]
        # offset = doc_text[0].i
        # mask_words_index.sort()
        # mask_words_index = [m - offset for m in mask_words_index]

        return mask_words_index, mask_words

    def _check_mask_candidate(
            self, mask_candidate, mask_word, all_entity_keywords=None, all_yake_keywords=None, keyword_ablate=False
    ):
        if all_yake_keywords is None:
            all_yake_keywords = []
        if all_entity_keywords is None:
            all_entity_keywords = []

        if keyword_ablate:
            if (mask_candidate not in all_entity_keywords
                    and not mask_candidate.is_punct
                    and not mask_candidate.pos_ == "PART"
                    and not _contains_punctuation(mask_candidate)
            ):
                return True
            else:
                return False

        if (mask_candidate not in all_entity_keywords  # 不在命名实体关键词列表
                and mask_candidate not in all_yake_keywords  # 不在yake关键词列表
                and mask_candidate not in mask_word  # todo: 不在已选掩码词列表mask_word?
                and not mask_candidate.is_punct  # 不是标点符号mask_cand.is_punct
                and not mask_candidate.pos_ == "PART"  # 不是语气助词mask_cand.pos_ == "PART"
                and not _contains_punctuation(mask_candidate)  # 不包含标点符号_contains_punct(mask_cand)
                and mask_candidate.text not in self.custom_keywords):  # 不在自定义message中，即self.custom_keywords
            return True
        else:
            return False
