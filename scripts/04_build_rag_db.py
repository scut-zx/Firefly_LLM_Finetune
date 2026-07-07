"""
流萤 RAG 知识库构建器
从 firefly-skill markdown 文件和爬取的 BWIKI 数据构建 ChromaDB 向量数据库

架构（参考 Yixuan 助手的 RAG 设计）：
- Collection 1: character_card — 角色结构化信息（profile, personality, relations）
- Collection 2: worldview — 世界观与背景知识（memory, BWIKI stories）

RAG 痛点分析（内嵌于代码注释）：
- RAG 单独使用时：事实准确但风格不匹配，base model 语调缺乏角色特征
- firefly-skill 作为 system prompt：风格改善但受限于模型跟随 prompt 的能力
- firefly-skill + Fine-tuning：风格融入权重，但缺乏实时事实检索
- firefly-skill + Fine-tuning + RAG：最优方案，RAG 处理"知道什么"（事实），微调处理"怎么说"（风格）
"""
import json
import os
import sys
import re
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
SKILL_DIR = PROJECT_ROOT / "reference_code" / "firefly-skill"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "firefly_knowledge.json"


def parse_markdown_sections(filepath):
    """解析 markdown 文件中的章节"""
    if not filepath.exists():
        return {}

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = {}
    current_section = "_header"
    current_content = []

    for line in content.split('\n'):
        if line.startswith('## '):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections


def extract_evidence_items(text):
    """提取 [verbatim], [artifact], [impression] 标记的内容"""
    items = []
    pattern = r'\[(verbatim|artifact|impression)\]\s*(.+?)(?=\n|$)'
    for match in re.finditer(pattern, text):
        items.append({
            "type": match.group(1),
            "content": match.group(2).strip()
        })
    return items


def build_knowledge_base():
    """从 firefly-skill 和爬取数据构建结构化知识库"""
    knowledge = {
        "character_card": {},
        "worldview_entries": [],
    }

    # --- 角色卡 ---
    # profile.md
    profile_sections = parse_markdown_sections(SKILL_DIR / "profile.md")
    for section_name, section_text in profile_sections.items():
        items = extract_evidence_items(section_text)
        if items:
            knowledge["character_card"][f"profile_{section_name}"] = items

    # personality.md
    personality_sections = parse_markdown_sections(SKILL_DIR / "personality.md")
    for section_name, section_text in personality_sections.items():
        items = extract_evidence_items(section_text)
        if items:
            knowledge["character_card"][f"personality_{section_name}"] = items

    # interaction.md
    interaction_sections = parse_markdown_sections(SKILL_DIR / "interaction.md")
    for section_name, section_text in interaction_sections.items():
        items = extract_evidence_items(section_text)
        if items:
            knowledge["character_card"][f"interaction_{section_name}"] = items

    # relations.md
    relations_sections = parse_markdown_sections(SKILL_DIR / "relations.md")
    for section_name, section_text in relations_sections.items():
        items = extract_evidence_items(section_text)
        if items:
            knowledge["character_card"][f"relations_{section_name}"] = items

    # --- 世界观条目 ---
    # memory.md - 按章节拆分
    memory_path = SKILL_DIR / "memory.md"
    if memory_path.exists():
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_content = f.read()

        # 按 ## 拆分 memory 章节
        sections = re.split(r'\n## ', memory_content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            # 取第一行作为标题
            lines = section.split('\n')
            title = lines[0].strip('# ').strip()
            content = '\n'.join(lines[1:]).strip()
            if content and len(content) > 100:
                knowledge["worldview_entries"].append({
                    "category": title[:50],
                    "content": content[:1000],  # 截断过长内容
                    "source": "firefly-skill/memory.md"
                })

    # BWIKI 故事
    bwiki_stories_path = RAW_DATA_DIR / "bwiki_stories.json"
    if bwiki_stories_path.exists():
        with open(bwiki_stories_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        for story in stories:
            if story.get("content") and len(story["content"]) > 50:
                knowledge["worldview_entries"].append({
                    "category": f"BWIKI {story.get('title', '故事')}",
                    "content": story["content"][:1000],
                    "source": "BWIKI"
                })

    # BWIKI 故事详细
    story_details_path = RAW_DATA_DIR / "bwiki_story_details.txt"
    if story_details_path.exists():
        with open(story_details_path, 'r', encoding='utf-8') as f:
            story_text = f.read().strip()
        if story_text and len(story_text) > 50:
            # 按段落拆分
            paragraphs = story_text.split('\n')
            chunk = ""
            for para in paragraphs:
                if len(chunk) + len(para) < 800:
                    chunk += para + '\n'
                else:
                    if chunk.strip():
                        knowledge["worldview_entries"].append({
                            "category": "BWIKI 角色详情",
                            "content": chunk.strip()[:1000],
                            "source": "BWIKI"
                        })
                    chunk = para + '\n'
            if chunk.strip():
                knowledge["worldview_entries"].append({
                    "category": "BWIKI 角色详情",
                    "content": chunk.strip()[:1000],
                    "source": "BWIKI"
                })

    # Moegirl 数据
    moegirl_path = RAW_DATA_DIR / "moegirl_full.json"
    if moegirl_path.exists():
        with open(moegirl_path, 'r', encoding='utf-8') as f:
            moegirl_data = json.load(f)
        if "sections" in moegirl_data:
            for title, content in moegirl_data["sections"].items():
                if content and len(content) > 100:
                    knowledge["worldview_entries"].append({
                        "category": f"Moegirl {title}",
                        "content": content[:1000],
                        "source": "Moegirl Wiki"
                    })

    return knowledge


def build_chromadb(knowledge):
    """构建 ChromaDB 向量数据库"""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ chromadb 未安装！请运行: pip install chromadb sentence-transformers")
        return False

    print("创建 ChromaDB 客户端...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # 嵌入模型（中英双语，与参考项目一致）
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ========================================
    # Collection 1: 角色卡 (character_card)
    # ========================================
    print("\n[1/2] 构建角色卡 (character_card)...")
    char_coll = client.get_or_create_collection(
        "character_card",
        embedding_function=embed_fn
    )

    char_docs = []
    char_metas = []
    char_ids = []

    for key, value in knowledge["character_card"].items():
        # 将 evidence items 合并为文本
        texts = []
        for item in value:
            tag = {"verbatim": "原句", "artifact": "事实", "impression": "解读"}.get(item["type"], "")
            texts.append(f"【{tag}】{item['content']}")

        combined = "\n".join(texts)
        if combined:
            char_docs.append(f"【{key}】\n{combined}")
            char_metas.append({"field": key, "type": "character_card"})
            char_ids.append(f"char_{key.replace(' ', '_').replace('/', '_')[:60]}")

    if char_docs:
        # 清空旧数据
        try:
            existing = char_coll.get()
            if existing and existing["ids"]:
                char_coll.delete(ids=existing["ids"])
        except Exception:
            pass

        char_coll.add(documents=char_docs, metadatas=char_metas, ids=char_ids)
        print(f"  ✅ 角色卡: {len(char_docs)} 条文档")

    # ========================================
    # Collection 2: 世界观 (worldview)
    # ========================================
    print("\n[2/2] 构建世界观 (worldview)...")
    world_coll = client.get_or_create_collection(
        "worldview",
        embedding_function=embed_fn
    )

    world_docs = [e['content'] for e in knowledge["worldview_entries"]]
    world_metas = [
        {"category": e['category'], "source": e.get('source', 'unknown')}
        for e in knowledge["worldview_entries"]
    ]
    world_ids = [f"world_{i}" for i in range(len(world_docs))]

    if world_docs:
        # 清空旧数据
        try:
            existing = world_coll.get()
            if existing and existing["ids"]:
                world_coll.delete(ids=existing["ids"])
        except Exception:
            pass

        world_coll.add(documents=world_docs, metadatas=world_metas, ids=world_ids)
        print(f"  ✅ 世界观: {len(world_docs)} 条文档")

    print(f"\n✅ ChromaDB 构建完成！")
    print(f"   存储位置: {CHROMA_DB_DIR}")
    print(f"   角色卡: {len(char_docs)} 条")
    print(f"   世界观: {len(world_docs)} 条")

    return True


def test_retrieval():
    """测试检索效果"""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    test_queries = [
        "流萤是谁？",
        "失熵症是什么？",
        "流萤和开拓者是什么关系？",
        "匹诺康尼发生了什么？",
        "萨姆装甲的功能是什么？",
    ]

    print("\n" + "=" * 60)
    print("RAG 检索测试")
    print("=" * 60)

    for query in test_queries:
        print(f"\n🔍 查询: {query}")

        try:
            char_coll = client.get_collection("character_card", embedding_function=embed_fn)
            char_results = char_coll.query(query_texts=[query], n_results=2)
            if char_results['documents'][0]:
                print(f"  [角色卡] {char_results['documents'][0][0][:100]}...")
        except Exception as e:
            print(f"  [角色卡] 检索失败: {e}")

        try:
            world_coll = client.get_collection("worldview", embedding_function=embed_fn)
            world_results = world_coll.query(query_texts=[query], n_results=2)
            if world_results['documents'][0]:
                print(f"  [世界观] {world_results['documents'][0][0][:100]}...")
        except Exception as e:
            print(f"  [世界观] 检索失败: {e}")


def main():
    print("=" * 60)
    print("流萤 RAG 知识库构建器")
    print("=" * 60)

    # ========================================
    # RAG 痛点分析 (文档化)
    # ========================================
    print("""
╔══════════════════════════════════════════════════════════════╗
║              RAG 痛点分析 — 产品设计角度                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 1. RAG 单独使用 (无微调无 skill):                                ║
║    优点: 事实准确（检索精确文本）                                  ║
║    缺点: 风格不匹配 — base model 用通用助手语调回答                ║
║    例子: 问"你是谁" → "我是Qwen3-4B，一个AI助手..." (OOC!)         ║
║                                                              ║
║ 2. firefly-skill (纯 system prompt):                          ║
║    优点: 有角色框架，语调有改善                                   ║
║    缺点: 受限于模型跟随 prompt 的能力；长对话中角色漂移             ║
║    例子: 短期能保持人设，超过10轮后语气趋于通用化                    ║
║                                                              ║
║ 3. firefly-skill + Fine-tuning (无 RAG):                      ║
║    优点: 风格值入权重，多轮对话稳定；情绪、关系处理自然             ║
║    缺点: 无法检索最新/详细事实；训练数据覆盖面有限                  ║
║    例子: 对于训练数据没覆盖到的冷门问题，模型可能编造事实            ║
║                                                              ║
║ 4. firefly-skill + Fine-tuning + RAG (本方案):                ║
║    优点: 风格稳定 + 事实准确，互补式增强                          ║
║    缺点: 复杂度增加；RAG 检索质量决定事实补充效果                  ║
║    核心洞察: RAG 处理"知道什么"(事实)，微调处理"怎么说"(风格)       ║
║                                                              ║
║ vs 单独 Skill: Skill 是静态规则，微调是"学会"的直觉              ║
║ vs 单独微调: 微调可能遗忘细节，RAG 提供实时检索                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # 构建知识库
    print("[1/3] 提取知识...")
    knowledge = build_knowledge_base()

    char_fields = len(knowledge["character_card"])
    world_entries = len(knowledge["worldview_entries"])
    print(f"  角色卡字段: {char_fields}")
    print(f"  世界观条目: {world_entries}")

    # 保存结构化知识（JSON格式，供后端加载）
    with open(KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 知识库 JSON: {KNOWLEDGE_PATH}")

    # 构建 ChromaDB
    print("\n[2/3] 构建 ChromaDB...")
    success = build_chromadb(knowledge)

    # 测试检索
    if success:
        print("\n[3/3] 测试检索...")
        test_retrieval()

    print("\n✅ RAG 知识库构建完成！")


if __name__ == "__main__":
    main()
