from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillManifest(BaseModel):
    """Server-owned instructions and capability policy, never an executable plugin."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    applicable_intents: list[str] = Field(default_factory=list)
    allowed_capabilities: list[str] = Field(default_factory=list)
    write_permission: bool = False
    confirmation_required: bool = True
    instruction_file: str = "instructions.md"


@dataclass(frozen=True)
class SkillPack:
    manifest: SkillManifest
    instructions: str
    root: Path


class SkillRegistry:
    """Loads a static allowlist of versioned skill packs from the application tree."""

    def __init__(self, root: Path, allowlist: set[str] | None = None) -> None:
        self.root = Path(root).resolve()
        self.allowlist = allowlist
        self._packs: dict[str, SkillPack] = {}

    def load(self) -> "SkillRegistry":
        packs: dict[str, SkillPack] = {}
        if not self.root.exists():
            self._packs = packs
            return self
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            if self.allowlist is not None and manifest.id not in self.allowlist:
                continue
            pack_root = manifest_path.parent.resolve()
            instruction_path = (pack_root / manifest.instruction_file).resolve()
            if pack_root not in instruction_path.parents:
                raise ValueError(f"skill instruction escapes its pack directory: {manifest.id}")
            instructions = instruction_path.read_text(encoding="utf-8").strip()
            if not instructions:
                raise ValueError(f"skill has empty instructions: {manifest.id}")
            if manifest.id in packs:
                raise ValueError(f"duplicate skill id: {manifest.id}")
            packs[manifest.id] = SkillPack(manifest, instructions, pack_root)
        self._packs = packs
        return self

    def get(self, skill_id: str) -> SkillPack:
        try:
            return self._packs[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown or disabled skill: {skill_id}") from exc

    def select_for_intent(self, intent: str) -> list[SkillPack]:
        return [
            pack for pack in self._packs.values()
            if intent in pack.manifest.applicable_intents
        ]

    def compose_instructions(self, skill_ids: list[str]) -> str:
        selected = [self.get(skill_id) for skill_id in dict.fromkeys(skill_ids)]
        return "\n\n".join(
            f"[Skill {pack.manifest.id}@{pack.manifest.version}]\n{pack.instructions}"
            for pack in selected
        )

    def describe(self) -> list[dict[str, Any]]:
        return [pack.manifest.model_dump() for pack in self._packs.values()]
