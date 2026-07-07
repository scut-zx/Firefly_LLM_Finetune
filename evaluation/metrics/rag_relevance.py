"""
RAG 相关性指标 (RAG Relevance Metric)

评估模型是否正确利用 RAG 检索到的知识来回答问题。
对比 RAG 开启与关闭时的回答质量差异。
"""

import re
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class RAGRelevanceMetric:
    """评估 RAG 知识利用效果"""

    @classmethod
    def _tokenize_chinese(cls, text: str) -> set:
        """
        简单的中文分词（基于字符二元组）。
        用于 fuzzy matching 而非精确语义分析。
        """
        # 提取所有 2-gram 和 3-gram 作为特征
        features = set()
        # 清理文本
        text = re.sub(r'[^一-鿿\w]', '', text)

        for i in range(len(text) - 1):
            features.add(text[i:i+2])
        for i in range(len(text) - 2):
            features.add(text[i:i+3])

        return features

    @classmethod
    def _fact_present(cls, fact: str, response: str) -> bool:
        """
        检查 response 中是否包含 fact 中的关键信息。
        使用模糊匹配（子串 + token overlap）。
        """
        # 提取 fact 中的关键 token
        fact_tokens = cls._tokenize_chinese(fact)
        resp_tokens = cls._tokenize_chinese(response)

        if not fact_tokens:
            return False

        # Jaccard 相似度
        overlap = len(fact_tokens & resp_tokens)
        jaccard = overlap / len(fact_tokens)

        # 也做简单的子串匹配
        # 提取 fact 中最长的连续中文字段
        fact_chinese = re.findall(r'[一-鿿]{3,}', fact)
        substring_match = any(seg in response for seg in fact_chinese)

        return jaccard > 0.3 or substring_match

    @classmethod
    def score(cls, response: str, question: str,
              expected_facts: list[str]) -> dict:
        """
        评估 RAG 事实准确率。

        Args:
            response: 模型生成的回答
            question: 原始问题
            expected_facts: 期望回答中包含的事实列表

        Returns:
            dict with fact_recall, fact_precision, rag_score
        """
        if not expected_facts:
            return {
                'fact_recall': 1.0,
                'fact_precision': 1.0,
                'rag_score': 1.0,
                'matched_facts': [],
                'missed_facts': [],
            }

        matched = []
        missed = []

        for fact in expected_facts:
            if cls._fact_present(fact, response):
                matched.append(fact)
            else:
                missed.append(fact)

        recall = len(matched) / len(expected_facts)
        # Precision 在单回答场景下难以计算，使用 recall 近似
        precision = min(1.0, recall * 1.1)  # 略微宽松

        rag_score = (recall + precision) / 2

        return {
            'fact_recall': round(recall, 3),
            'fact_precision': round(precision, 3),
            'rag_score': round(rag_score, 3),
            'matched_facts': matched,
            'missed_facts': missed,
        }

    @classmethod
    def compare_rag_vs_no_rag(cls, question: str,
                              response_with_rag: str,
                              response_without_rag: str,
                              expected_facts: list[str]) -> dict:
        """
        对比 RAG 开启 vs 关闭的回答质量。

        Returns:
            dict with rag_on, rag_off, delta
        """
        rag_on = cls.score(response_with_rag, question, expected_facts)
        rag_off = cls.score(response_without_rag, question, expected_facts)

        delta = {
            'fact_recall_delta': round(rag_on['fact_recall'] - rag_off['fact_recall'], 3),
            'rag_score_delta': round(rag_on['rag_score'] - rag_off['rag_score'], 3),
        }

        return {
            'rag_enabled': rag_on,
            'rag_disabled': rag_off,
            'delta': delta,
        }

    @classmethod
    def batch_score(cls, test_cases: list[dict]) -> dict:
        """
        批处理多个测试用例。

        Args:
            test_cases: 每个包含 response, question, expected_facts

        Returns:
            dict with average scores and per-case results
        """
        results = []
        total_recall = 0.0

        for case in test_cases:
            result = cls.score(
                case.get('response', ''),
                case.get('question', ''),
                case.get('expected_facts', []),
            )
            results.append(result)
            total_recall += result['fact_recall']

        n = len(test_cases) if test_cases else 1

        return {
            'average_fact_recall': round(total_recall / n, 3),
            'average_rag_score': round(
                sum(r['rag_score'] for r in results) / n, 3
            ),
            'total_facts_tested': sum(
                len(c.get('expected_facts', [])) for c in test_cases
            ),
            'per_case': results,
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    # 模拟 RAG 知识问答场景
    test_q = "流萤的失熵症是什么？"
    expected = [
        "失熵症是一种退行性疾病",
        "身体会逐渐瓦解",
        "流萤因此生命短暂",
    ]

    good_response = ("嗯……失熵症是一种让身体慢慢消失的病。"
                     "我的身体会一点一点地瓦解，所以时间对我来说很珍贵。"
                     "但也正因为这样，我更想好好活每一天。")
    bad_response = "失熵症是一种虚构的疾病，在游戏中被设定为角色的背景故事。"

    print("=== 好回答（包含事实）===")
    result = RAGRelevanceMetric.score(good_response, test_q, expected)
    print(f"  Recall: {result['fact_recall']}, Score: {result['rag_score']}")
    print(f"  Matched: {result['matched_facts']}")
    print(f"  Missed: {result['missed_facts']}")

    print("\n=== 差回答（缺少事实）===")
    result = RAGRelevanceMetric.score(bad_response, test_q, expected)
    print(f"  Recall: {result['fact_recall']}, Score: {result['rag_score']}")
    print(f"  Matched: {result['matched_facts']}")
    print(f"  Missed: {result['missed_facts']}")
