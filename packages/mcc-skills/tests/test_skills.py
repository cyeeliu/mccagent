"""Skill 与 Slash Command 加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_skills import (
    Skill,
    SkillLoader,
    SkillManifest,
    SkillRouter,
    SlashCommand,
    SlashRegistry,
    load_commands_merged,
    load_skills_merged,
    parse_frontmatter,
    resolve_slash,
)


class TestParseFrontmatter:
    @pytest.mark.parametrize(
        "text,expect_meta,body",
        [
            ("---\nname: x\ndescription: y\n---\nbody", {"name": "x", "description": "y"}, "body"),
            ("no frontmatter here", {}, "no frontmatter here"),
            ("---\nname: x\n---\nhello", {"name": "x"}, "hello"),
        ],
        ids=["normal", "no-frontmatter", "minimal"],
    )
    def test_parse(self, text, expect_meta, body):
        meta, b = parse_frontmatter(text)
        assert meta == expect_meta
        assert b == body

    def test_no_closing_fence(self):
        text = "---\nname: x\nbody without close"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_yaml_error_falls_back_to_empty(self):
        meta, body = parse_frontmatter("---\n: : :\n---\nbody")
        assert meta == {}
        assert body == "body"

    def test_multiline_description(self):
        text = "---\nname: x\ndescription: |\n  line1\n  line2\n---\nbody"
        meta, _ = parse_frontmatter(text)
        assert meta["name"] == "x"
        assert "line1" in meta["description"]
        assert "line2" in meta["description"]


def _write_skill(root: Path, name: str, content: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")


class TestSkillLoader:
    def test_empty_dir(self, tmp_path):
        assert SkillLoader(tmp_path / "skills").skills == {}

    def test_nonexistent_dir(self, tmp_path):
        assert SkillLoader(tmp_path / "nope").skills == {}

    def test_loads_skill(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(
            root, "pr", "---\nname: pr\ndescription: review PRs\n---\n# PR Reviewer\nsteps"
        )
        sk = SkillLoader(root).get("pr")
        assert sk is not None
        assert sk.manifest.description == "review PRs"
        assert "PR Reviewer" in sk.content

    def test_name_defaults_to_dir_name(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root, "my-skill", "---\ndescription: desc\n---\nbody")
        loader = SkillLoader(root)
        assert loader.get("my-skill").manifest.name == "my-skill"

    def test_description_defaults_to_body_first_line(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root, "x", "---\nname: x\n---\n# Heading\nrest")
        assert SkillLoader(root).get("x").manifest.description == "Heading"

    def test_triggers_model_requires_parsed(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(
            root, "x",
            "---\nname: x\ndescription: d\ntriggers:\n  - /review\n  - review this\n"
            "model: claude-opus\nrequires_tools:\n  - read_file\n  - run_bash\n---\nbody",
        )
        m = SkillLoader(root).get("x").manifest
        assert m.triggers == ["/review", "review this"]
        assert m.model == "claude-opus"
        assert m.requires_tools == ["read_file", "run_bash"]

    def test_skips_dir_without_skill_file(self, tmp_path):
        root = tmp_path / "skills"
        (root / "empty").mkdir(parents=True)
        _write_skill(root, "real", "---\nname: real\ndescription: d\n---\nbody")
        assert list(SkillLoader(root).skills) == ["real"]

    def test_names_sorted(self, tmp_path):
        root = tmp_path / "skills"
        for n in ["b", "a", "c"]:
            _write_skill(root, n, f"---\nname: {n}\ndescription: d\n---\nbody")
        loader = SkillLoader(root)
        assert loader.names() == ["a", "b", "c"]


class TestSlashRegistry:
    @staticmethod
    def _write_cmd(root: Path, name: str, content: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.md").write_text(content, encoding="utf-8")

    def test_loads_command(self, tmp_path):
        root = tmp_path / "commands"
        self._write_cmd(
            root, "test", "---\ncommand: /test\ndescription: run tests\n---\nexecute pytest"
        )
        reg = SlashRegistry(root)
        assert reg.commands["/test"].body == "execute pytest"

    def test_command_defaults_to_slash_stem(self, tmp_path):
        root = tmp_path / "commands"
        self._write_cmd(root, "build", "---\ndescription: build\n---\ncargo build")
        assert "/build" in SlashRegistry(root).commands

    def test_resolve_hit(self, tmp_path):
        root = tmp_path / "commands"
        self._write_cmd(root, "test", "---\ncommand: /test\ndescription: d\n---\nrun pytest")
        assert SlashRegistry(root).resolve("/test") == "run pytest"

    def test_resolve_with_args(self, tmp_path):
        root = tmp_path / "commands"
        self._write_cmd(root, "test", "---\ncommand: /test\ndescription: d\n---\nrun pytest")
        result = SlashRegistry(root).resolve("/test -xvs")
        assert "run pytest" in result
        assert "-xvs" in result

    @pytest.mark.parametrize("inp", ["/unknown", "", "   "], ids=["unknown", "empty", "blank"])
    def test_resolve_miss(self, tmp_path, inp):
        root = tmp_path / "commands"
        self._write_cmd(root, "test", "---\ncommand: /test\ndescription: d\n---\nbody")
        assert SlashRegistry(root).resolve(inp) is None


class TestSkillRouter:
    def test_catalog_empty(self):
        assert SkillRouter({}).catalog_for_prompt() == ""

    def test_catalog_lists_skills(self):
        skills = {
            "a": Skill(SkillManifest(name="a", description="desc a"), "c", Path()),
            "b": Skill(SkillManifest(name="b", description="desc b"), "c", Path()),
        }
        catalog = SkillRouter(skills).catalog_for_prompt()
        assert "**a**" in catalog
        assert "**b**" in catalog
        assert "load_skill" in catalog

    def test_match_slash_hit(self):
        sk = Skill(SkillManifest(name="r", description="d", triggers=["/review"]), "c", Path())
        assert SkillRouter({"r": sk}).match_slash("/review") is sk

    def test_match_slash_miss(self):
        sk = Skill(SkillManifest(name="r", description="d", triggers=["/review"]), "c", Path())
        assert SkillRouter({"r": sk}).match_slash("/other") is None

    def test_match_slash_empty(self):
        assert SkillRouter({}).match_slash("/x") is None


class TestMerge:
    def test_skills_project_overrides_home(self, tmp_path):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        _write_skill(home, "x", "---\nname: x\ndescription: home desc\n---\nhome body")
        _write_skill(proj, "x", "---\nname: x\ndescription: proj desc\n---\nproj body")
        assert load_skills_merged(home, proj)["x"].manifest.description == "proj desc"

    def test_skills_home_only(self, tmp_path):
        home = tmp_path / "home"
        _write_skill(home, "x", "---\nname: x\ndescription: d\n---\nbody")
        assert "x" in load_skills_merged(home, None)

    def test_skills_project_only(self, tmp_path):
        proj = tmp_path / "proj"
        _write_skill(proj, "x", "---\nname: x\ndescription: d\n---\nbody")
        assert "x" in load_skills_merged(None, proj)

    def test_commands_project_overrides_home(self, tmp_path):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        home.mkdir()
        proj.mkdir()
        (home / "t.md").write_text("---\ncommand: /t\ndescription: h\n---\nhome body")
        (proj / "t.md").write_text("---\ncommand: /t\ndescription: p\n---\nproj body")
        assert load_commands_merged(home, proj)["/t"].body == "proj body"


class TestResolveSlash:
    def test_hit(self):
        cmds = {"/t": SlashCommand(name="/t", body="run", description="d")}
        assert resolve_slash(cmds, "/t") == "run"

    def test_miss(self):
        assert resolve_slash({}, "/t") is None

    def test_with_args(self):
        cmds = {"/t": SlashCommand(name="/t", body="run", description="d")}
        result = resolve_slash(cmds, "/t extra")
        assert "run" in result
        assert "extra" in result
