"""
崩坏星穹铁道 - 流萤角色资料爬虫
从 BWIKI 和 Moegirl Wiki 爬取流萤的所有信息，分类整理
改编自 reference_code/zzz-yixuan-dataset/crawl_yixuan.py
"""
import re
import json
import html as html_module
import os
import sys
import time
import requests
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_page(url, retries=3):
    """抓取页面，带重试"""
    for attempt in range(retries):
        try:
            print(f"  抓取: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = 'utf-8'
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"  HTTP {resp.status_code}, 重试 {attempt+1}/{retries}...")
                time.sleep(2)
        except Exception as e:
            print(f"  抓取失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return None


def clean_text(text):
    """清理 HTML 标签和空白"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html_module.unescape(text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_section(html, section_name, next_sections=None):
    """提取指定标题下的内容（BWIKI mw-headline 格式）"""
    pattern = rf'<span class="mw-headline"[^>]*>{re.escape(section_name)}</span>'
    match = re.search(pattern, html)
    if not match:
        return ""

    start = match.start()

    # 找下一个标题位置
    end = len(html)
    if next_sections:
        for ns in next_sections:
            ns_escaped = re.escape(ns)
            next_pattern = rf'<span class="mw-headline"[^>]*>{ns_escaped}</span>'
            next_match = re.search(next_pattern, html[start+10:])
            if next_match:
                end = start + 10 + next_match.start()
                break
    else:
        # 找任意下一个标题
        next_match = re.search(r'<span class="mw-headline"', html[start+10:])
        if next_match:
            end = start + 10 + next_match.start()

    section = html[start:end]
    return clean_text(section)


def extract_basic_info(html):
    """提取基础信息表（BWIKI 右侧 infobox）"""
    info = {}
    # 方法1: 找 th/td 对
    pairs = re.findall(r'<th[^>]*>([^<]+)</th>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    for k, v in pairs:
        k_clean = clean_text(k)
        v_clean = clean_text(v)
        if k_clean and v_clean and len(k_clean) < 20:
            info[k_clean] = v_clean

    # 方法2: 如果方法1没找到足够信息，尝试找 infobox 里的行
    if len(info) < 3:
        infobox_pattern = re.findall(
            r'<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*</tr>',
            html, re.DOTALL
        )
        for v1, v2 in infobox_pattern:
            k_clean = clean_text(v1)
            v_clean = clean_text(v2)
            if k_clean and v_clean and len(k_clean) < 20:
                info[k_clean] = v_clean

    return info


def extract_skills(html):
    """提取技能信息"""
    skills = []
    skill_tables = re.findall(
        r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>',
        html, re.DOTALL
    )
    for table in skill_tables[:10]:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) >= 2:
                name = clean_text(cells[0])
                desc = clean_text(' '.join(cells[1:]))
                if name and desc and len(name) < 30 and len(desc) > 10:
                    skills.append({"name": name, "description": desc})
    return skills


def extract_voice_lines(html):
    """提取语音台词（从「」引号内容中提取，过滤技能名）"""
    voices = []
    quoted = re.findall(r'「([^」]{5,300})」', html)

    skill_keywords = ['攻击', '技', '强化', '终结', '普攻', '闪避', '支援', '连携',
                      '战技', '天赋', '秘技', '行迹', '星魂']

    for q in quoted:
        q_clean = re.sub(r'<[^>]+>', '', q).strip()
        if not q_clean or len(q_clean) < 5:
            continue
        # 排除技能名
        if any(kw in q_clean for kw in skill_keywords) and len(q_clean) < 15:
            continue
        if q_clean.startswith('强化') or q_clean.startswith('终结') or q_clean.startswith('普通'):
            continue
        # 只要包含中文标点的才算台词
        if any(p in q_clean for p in '。，！？…—'):
            voices.append({"scene": "台词", "text": q_clean})

    # 去重
    seen = set()
    unique_voices = []
    for v in voices:
        if v["text"] not in seen:
            seen.add(v["text"])
            unique_voices.append(v)

    return unique_voices


def extract_character_stories(html):
    """提取角色故事"""
    stories = []
    # BWIKI 角色故事通常按"角色故事一/二/三/四"分节
    for i in range(1, 6):
        story_text = extract_section(html, f"角色故事{i}")
        if story_text:
            stories.append({"title": f"角色故事{i}", "content": story_text})

    # 如果没有找到，尝试"角色详情"等替代
    if not stories:
        alt_names = ["角色详情", "角色资料", "角色背景", "详细情报"]
        for alt in alt_names:
            story_text = extract_section(html, alt)
            if story_text:
                stories.append({"title": alt, "content": story_text})
                break

    return stories


def extract_moegirl_sections(html):
    """从 Moegirl Wiki 提取内容（格式与 BWIKI 不同）"""
    sections = {}

    # Moegirl 使用 h2/h3 标签
    h2_pattern = re.findall(
        r'<h2[^>]*>\s*<span[^>]*class="mw-headline"[^>]*>(.*?)</span>\s*</h2>(.*?)(?=<h2|$)',
        html, re.DOTALL
    )

    for title, content in h2_patterns:
        title_clean = clean_text(title)
        content_clean = clean_text(content)
        if title_clean and content_clean and len(content_clean) > 50:
            sections[title_clean] = content_clean[:5000]  # 截断过长内容

    return sections


# ============================================================
# BWIKI 爬取
# ============================================================
def crawl_bwiki():
    """爬取 BWIKI 崩坏星穹铁道流萤页面"""
    print("=" * 60)
    print("[BWIKI] 爬取流萤角色资料")
    print("=" * 60)

    url = "https://wiki.biligame.com/sr/流萤"
    html = fetch_page(url)
    if not html:
        print("BWIKI 爬取失败！尝试备用URL...")
        url_alt = "https://wiki.biligame.com/sr/%E6%B5%81%E8%90%A4"
        html = fetch_page(url_alt)
        if not html:
            print("备用URL也失败了")
            return None

    print(f"页面大小: {len(html)} 字符")

    # 提取信息
    print("\n[1/6] 提取基础信息...")
    basic_info = extract_basic_info(html)
    print(f"  基础信息: {len(basic_info)} 项")

    print("[2/6] 提取角色故事...")
    stories = extract_character_stories(html)
    print(f"  角色故事: {len(stories)} 篇")

    print("[3/6] 提取技能...")
    skills = extract_skills(html)
    print(f"  技能: {len(skills)} 个")

    print("[4/6] 提取语音台词...")
    voices = extract_voice_lines(html)
    print(f"  语音台词: {len(voices)} 条")

    print("[5/6] 提取额外信息...")
    # 提取各种可能有用的章节
    extra_sections = {}
    for section_name in ["角色简介", "角色经历", "角色关系", "角色考据", "光锥推荐", "遗器推荐"]:
        content = extract_section(html, section_name)
        if content:
            extra_sections[section_name] = content
            print(f"  找到章节: {section_name} ({len(content)} 字)")

    print("[6/6] 提取角色故事详细...")
    story_details = extract_section(html, "角色故事", ["语音台词", "相关视频", "角色考据"])
    if not story_details:
        story_details = extract_section(html, "故事", ["语音台词", "相关视频"])

    # 组装
    bwiki_data = {
        "source": url,
        "character": {
            "name": "流萤",
            "game": "崩坏：星穹铁道",
        },
        "basic_info": basic_info,
        "stories": stories,
        "story_details": story_details,
        "skills": skills,
        "voices": voices,
        "extra_sections": extra_sections,
    }

    return bwiki_data


# ============================================================
# Moegirl Wiki 爬取
# ============================================================
def crawl_moegirl():
    """爬取 Moegirl Wiki 流萤页面"""
    print("\n" + "=" * 60)
    print("[Moegirl Wiki] 爬取流萤角色资料")
    print("=" * 60)

    url = "https://zh.moegirl.org.cn/zh-hans/流萤"
    html = fetch_page(url)
    if not html:
        print("Moegirl Wiki 爬取失败！")
        return None

    print(f"页面大小: {len(html)} 字符")

    # 提取信息表
    print("\n[1/3] 提取基础信息...")
    basic_info = extract_basic_info(html)
    print(f"  基础信息: {len(basic_info)} 项")

    print("[2/3] 提取各章节...")
    sections = extract_moegirl_sections(html)
    print(f"  章节: {list(sections.keys())}")

    print("[3/3] 提取语音...")
    voices = extract_voice_lines(html)
    print(f"  语音: {len(voices)} 条")

    moegirl_data = {
        "source": url,
        "character": {
            "name": "流萤",
            "game": "崩坏：星穹铁道",
        },
        "basic_info": basic_info,
        "sections": sections,
        "voices": voices,
    }

    return moegirl_data


# ============================================================
# 保存数据
# ============================================================
def save_data(bwiki_data, moegirl_data):
    """保存所有爬取数据"""
    print("\n" + "=" * 60)
    print("保存数据...")
    print("=" * 60)

    # BWIKI 数据
    if bwiki_data:
        # 分类保存
        with open(OUTPUT_DIR / "bwiki_basic_info.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data["basic_info"], f, ensure_ascii=False, indent=2)
        print(f"  ✅ bwiki_basic_info.json ({len(bwiki_data['basic_info'])} 项)")

        with open(OUTPUT_DIR / "bwiki_stories.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data["stories"], f, ensure_ascii=False, indent=2)
        print(f"  ✅ bwiki_stories.json ({len(bwiki_data['stories'])} 篇)")

        with open(OUTPUT_DIR / "bwiki_story_details.txt", 'w', encoding='utf-8') as f:
            f.write(bwiki_data.get("story_details", ""))
        print(f"  ✅ bwiki_story_details.txt ({len(bwiki_data.get('story_details', ''))} 字)")

        with open(OUTPUT_DIR / "bwiki_skills.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data["skills"], f, ensure_ascii=False, indent=2)
        print(f"  ✅ bwiki_skills.json ({len(bwiki_data['skills'])} 个)")

        with open(OUTPUT_DIR / "bwiki_voices.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data["voices"], f, ensure_ascii=False, indent=2)
        print(f"  ✅ bwiki_voices.json ({len(bwiki_data['voices'])} 条)")

        with open(OUTPUT_DIR / "bwiki_extra.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data.get("extra_sections", {}), f, ensure_ascii=False, indent=2)

        # 完整数据
        with open(OUTPUT_DIR / "bwiki_full.json", 'w', encoding='utf-8') as f:
            json.dump(bwiki_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ bwiki_full.json (完整数据)")

    # Moegirl 数据
    if moegirl_data:
        with open(OUTPUT_DIR / "moegirl_full.json", 'w', encoding='utf-8') as f:
            json.dump(moegirl_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ moegirl_full.json")

        with open(OUTPUT_DIR / "moegirl_voices.json", 'w', encoding='utf-8') as f:
            json.dump(moegirl_data["voices"], f, ensure_ascii=False, indent=2)
        print(f"  ✅ moegirl_voices.json ({len(moegirl_data['voices'])} 条)")

    # 合并所有语音并去重
    all_voices = []
    seen = set()
    if bwiki_data:
        for v in bwiki_data["voices"]:
            if v["text"] not in seen:
                seen.add(v["text"])
                all_voices.append(v)
    if moegirl_data:
        for v in moegirl_data["voices"]:
            if v["text"] not in seen:
                seen.add(v["text"])
                all_voices.append(v)

    with open(OUTPUT_DIR / "all_voices_merged.json", 'w', encoding='utf-8') as f:
        json.dump(all_voices, f, ensure_ascii=False, indent=2)
    print(f"  ✅ all_voices_merged.json ({len(all_voices)} 条，已去重)")

    # 生成文本格式摘要
    summary_lines = [
        "=" * 60,
        "流萤 角色资料爬取摘要",
        "=" * 60,
        f"爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if bwiki_data:
        summary_lines.extend([
            "--- BWIKI ---",
            f"基础信息项数: {len(bwiki_data['basic_info'])}",
            f"角色故事: {len(bwiki_data['stories'])} 篇",
            f"技能: {len(bwiki_data['skills'])} 个",
            f"语音台词: {len(bwiki_data['voices'])} 条",
        ])
        if bwiki_data.get("extra_sections"):
            summary_lines.append(f"额外章节: {', '.join(bwiki_data['extra_sections'].keys())}")
    if moegirl_data:
        summary_lines.extend([
            "",
            "--- Moegirl Wiki ---",
            f"章节数: {len(moegirl_data['sections'])}",
            f"语音台词: {len(moegirl_data['voices'])} 条",
        ])
    summary_lines.append(f"\n总计去重语音: {len(all_voices)} 条")

    summary = "\n".join(summary_lines)
    with open(OUTPUT_DIR / "crawl_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  ✅ crawl_summary.txt")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("崩坏星穹铁道 - 流萤角色资料爬虫")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 1. 爬取 BWIKI
    bwiki_data = crawl_bwiki()

    # 礼貌延迟，避免被封
    time.sleep(3)

    # 2. 爬取 Moegirl Wiki
    moegirl_data = crawl_moegirl()

    # 3. 保存
    if bwiki_data or moegirl_data:
        save_data(bwiki_data, moegirl_data)
        print("\n✅ 爬取完成！")
    else:
        print("\n❌ 所有来源爬取失败！请检查网络连接。")
        print("提示：如果 BWIKI 反爬，可以尝试：")
        print("  1. 在浏览器中打开页面，查看实际URL")
        print("  2. 检查是否需要添加 Cookie")
        print("  3. 使用 Playwright 替代 requests（参考 reference_code 中的方法）")
        sys.exit(1)

    # 打印预览
    if bwiki_data:
        print(f"\n=== BWIKI 基础信息 ===")
        for k, v in list(bwiki_data["basic_info"].items())[:10]:
            print(f"  {k}: {v}")

        print(f"\n=== 语音台词示例 ===")
        for v in bwiki_data["voices"][:5]:
            print(f"  [{v['scene']}] {v['text'][:80]}...")


if __name__ == "__main__":
    main()
