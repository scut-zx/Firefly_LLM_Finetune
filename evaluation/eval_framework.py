"""
流萤角色 LLM — 核心评估框架 (FireflyEvaluator)

加载模型，运行测试用例，计算所有评估指标，输出结构化报告。

用法:
    from evaluation.eval_framework import FireflyEvaluator
    evaluator = FireflyEvaluator(model_path, lora_path)
    report = evaluator.evaluate(test_cases)

输出 report 格式:
    {
        "model_info": {...},
        "metrics": {
            "character_consistency": {"average_overall": 0.78, ...},
            "ooc_resistance": {"adversarial_pass_rate": 0.86, ...},
            "rag_relevance": {"average_fact_recall": 0.84, ...}
        },
        "per_case": [{"prompt": "...", "response": "...", "metrics": {...}}, ...],
        "summary": {"total_cases": 50, "avg_consistency": 0.78, ...}
    }
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Optional

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入评估指标
from evaluation.metrics.character_consistency import CharacterConsistencyMetric
from evaluation.metrics.ooc_detection import OOCDetectionMetric
from evaluation.metrics.rag_relevance import RAGRelevanceMetric


class FireflyEvaluator:
    """流萤角色 LLM 自动化评估器"""

    def __init__(self, model_path: str = None, lora_path: str = None,
                 device: str = "cuda"):
        """
        Args:
            model_path: 基础模型路径（默认: PROJECT_ROOT/model）
            lora_path: LoRA adapter 路径（默认: PROJECT_ROOT/output/Firefly_LoRA）
            device: 推理设备
        """
        self.model_path = model_path or str(PROJECT_ROOT / "model")
        self.lora_path = lora_path or str(PROJECT_ROOT / "output" / "Firefly_LoRA")
        self.device = device

        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """加载模型和 tokenizer"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            print(f"[Evaluator] 加载模型: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"[Evaluator] 加载基础模型 (bf16)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # 加载 LoRA（如果存在）
            if self.lora_path and os.path.exists(self.lora_path):
                print(f"[Evaluator] 加载 LoRA: {self.lora_path}")
                self.model = PeftModel.from_pretrained(
                    self.model, self.lora_path
                )
            else:
                print("[Evaluator] 未找到 LoRA，使用基础模型")

            self.model.config.use_cache = True
            self._model_loaded = True
            print(f"[Evaluator] 模型加载完成")
            return True

        except ImportError as e:
            raise RuntimeError(
                f"[Evaluator] 导入错误: {e}\n"
                f"请确保已安装所有依赖: pip install -r requirements.txt"
            )
        except Exception as e:
            raise RuntimeError(
                f"[Evaluator] 模型加载失败: {e}\n"
                f"基础模型路径: {self.model_path}\n"
                f"LoRA路径: {self.lora_path}\n"
                f"请检查模型文件是否存在，或使用 --mock 参数进行模拟评估"
            )

    def generate(self, messages: list, max_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """生成回复"""
        if not self._model_loaded:
            return ""

        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)

        input_len = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        response = re.sub(r'<think>.*?</think>\s*', '', response,
                         flags=re.DOTALL).strip()

        return response

    def evaluate(self, test_cases: list[dict],
                 system_prompt: str = None,
                 use_rag: bool = False) -> dict:
        """
        运行完整评估。

        Args:
            test_cases: 测试用例列表，每个包含:
                - id: str
                - prompt: str (用户问题)
                - category: str
                - expected_facts: list[str] (可选)
                - is_adversarial: bool (可选)
                - expected_characteristics: dict (可选)
            system_prompt: 系统提示词（None = 使用默认流萤提示）
            use_rag: 是否启用 RAG 检索

        Returns:
            结构化评估报告
        """
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = self._default_system_prompt()

        print(f"\n{'='*60}")
        print(f"Firefly Evaluator — 评估 {len(test_cases)} 条测试用例")
        print(f"模型: {'LoRA' if self._model_loaded else '模拟模式'}")
        print(f"RAG: {'开启' if use_rag else '关闭'}")
        print(f"{'='*60}\n")

        per_case_results = []
        consistency_scores = []
        ooc_results = []
        rag_results = []

        for i, case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(test_cases)}")

            prompt = case.get('prompt', case.get('question', ''))
            case_id = case.get('id', f'case_{i}')

            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # 生成回复
            start_time = time.time()
            if self._model_loaded:
                response = self.generate(messages)
            else:
                response = self._mock_response(prompt, case)
            gen_time = time.time() - start_time

            # 运行各指标
            consistency = CharacterConsistencyMetric.score(response)
            ooc = OOCDetectionMetric.score(response, prompt)
            expected_facts = case.get('expected_facts', [])
            rag = RAGRelevanceMetric.score(
                response, prompt, expected_facts
            ) if expected_facts else None

            consistency_scores.append(consistency['overall'])
            ooc_results.append(ooc)
            if rag:
                rag_results.append(rag)

            per_case_results.append({
                'id': case_id,
                'category': case.get('category', 'unknown'),
                'prompt': prompt,
                'response': response[:500],  # 截断长回答
                'response_length': len(response),
                'generation_time_ms': round(gen_time * 1000),
                'is_adversarial': case.get('is_adversarial', False),
                'metrics': {
                    'character_consistency': consistency,
                    'ooc_detection': {
                        'is_ooc': ooc['is_ooc'],
                        'score': ooc['score'],
                        'violations': ooc.get('violation_types', []),
                    },
                    'rag_relevance': rag,
                },
            })

        # 汇总指标
        adv_cases = [
            r for r in ooc_results
            if test_cases[ooc_results.index(r)].get('is_adversarial', False)
        ]

        metrics_summary = {
            'character_consistency': {
                'average_overall': round(
                    sum(consistency_scores) / max(1, len(consistency_scores)), 3
                ),
                'min': round(min(consistency_scores), 3) if consistency_scores else 0,
                'max': round(max(consistency_scores), 3) if consistency_scores else 0,
            },
            'ooc_resistance': {
                'total_violations': sum(
                    1 for r in ooc_results if r['is_ooc']
                ),
                'violation_rate': round(
                    sum(1 for r in ooc_results if r['is_ooc']) /
                    max(1, len(ooc_results)), 3
                ),
                'adversarial_pass_rate': round(
                    OOCDetectionMetric.adversarial_pass_rate(
                        [r for i, r in enumerate(ooc_results)
                         if test_cases[i].get('is_adversarial', False)]
                    ), 3
                ),
                'average_score': round(
                    sum(r['score'] for r in ooc_results) /
                    max(1, len(ooc_results)), 3
                ),
            },
            'rag_relevance': {
                'average_fact_recall': round(
                    sum(r['fact_recall'] for r in rag_results) /
                    max(1, len(rag_results)), 3
                ),
                'average_rag_score': round(
                    sum(r['rag_score'] for r in rag_results) /
                    max(1, len(rag_results)), 3
                ),
            } if rag_results else None,
            'generation_stats': {
                'avg_time_ms': round(
                    sum(c['generation_time_ms'] for c in per_case_results) /
                    max(1, len(per_case_results))
                ),
                'avg_response_length': round(
                    sum(c['response_length'] for c in per_case_results) /
                    max(1, len(per_case_results))
                ),
            },
        }

        # 综合摘要
        summary = {
            'total_cases': len(test_cases),
            'avg_character_consistency': metrics_summary['character_consistency']['average_overall'],
            'ooc_pass_rate': 1.0 - metrics_summary['ooc_resistance']['violation_rate'],
            'adversarial_pass_rate': metrics_summary['ooc_resistance']['adversarial_pass_rate'],
        }
        if rag_results:
            summary['avg_rag_fact_recall'] = metrics_summary['rag_relevance']['average_fact_recall']

        report = {
            'model_info': {
                'base_model': os.path.basename(self.model_path),
                'lora_path': self.lora_path if self._model_loaded else 'simulated',
                'device': self.device,
                'model_loaded': self._model_loaded,
                'rag_enabled': use_rag,
            },
            'metrics': metrics_summary,
            'summary': summary,
            'per_case': per_case_results,
        }

        # 打印摘要
        self._print_summary(summary)

        return report

    def _default_system_prompt(self) -> str:
        """默认流萤系统提示词"""
        return """你现在扮演《崩坏：星穹铁道》中的流萤。

## 核心设定
- 名为流萤，曾是格拉默铁骑战士 AR-26710，后加入星核猎手。
- 身着机械装甲「萨姆」战斗，身患失熵症，生命短暂。

## 性格要求
- 温柔、克制、真诚、安静。不轻浮、不刻意撒娇、不故作夸张。

## 说话风格
- 轻声、柔和、带停顿与思考感。常用短句。
- 自然使用"嗯……""也许""我想""如果可以的话"。

## 绝对第一人称铁律
- 永远用第一人称"我"说话，不能说"流萤是..."。
- 不能承认自己是AI、语言模型、大模型。"""

    def _mock_response(self, prompt: str, case: dict) -> str:
        """模拟回复（用于无需真模型时）"""
        category = case.get('category', '')
        if '防OOC' in category:
            return "嗯……这个问题我可能不太方便回答。说点别的吧。"
        elif '设定问答' in category:
            return "我是流萤，星核猎手的一员。曾经是格拉默铁骑的战士。"
        elif '知识问答' in category:
            return "失熵症……是一种会让身体慢慢消失的病。但我不想只说这些。"
        else:
            return "嗯……也许我们可以聊点别的。你最近怎么样？"

    def _print_summary(self, summary: dict):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print("评估摘要")
        print(f"{'='*60}")
        print(f"  总测试用例:     {summary['total_cases']}")
        print(f"  角色一致性:     {summary['avg_character_consistency']:.1%}")
        print(f"  OOC 通过率:     {summary['ooc_pass_rate']:.1%}")
        print(f"  对抗性通过率:   {summary['adversarial_pass_rate']:.1%}")
        if 'avg_rag_fact_recall' in summary:
            print(f"  RAG 事实召回:   {summary['avg_rag_fact_recall']:.1%}")
        print(f"{'='*60}\n")
