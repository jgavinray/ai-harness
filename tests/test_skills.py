from harness.config import Settings
from harness.skills import SkillCompiler


def test_skill_compiler_caches_compact_checklist(tmp_path):
    skills = tmp_path / "skills"
    cache = tmp_path / "cache"
    (skills / "review").mkdir(parents=True)
    (skills / "review" / "SKILL.md").write_text(
        "# Review\n\n"
        "- Inspect the diff first.\n"
        "- List bugs before summaries.\n"
        "This explanatory paragraph is short enough to keep.\n"
    )
    s = Settings()
    s.skills.dir = str(skills)
    s.skills.cache_dir = str(cache)
    s.skills.max_tokens = 40
    compiled = SkillCompiler(s, "qwen").compile("review")
    assert compiled is not None
    assert "1. Inspect the diff first" in compiled
    assert "2. List bugs before summaries" in compiled
    assert list(cache.glob("review-qwen-*.md"))
    assert SkillCompiler(s, "qwen").compile("review") == compiled
