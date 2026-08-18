from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence


MAJOR_MATCH_SCORE = 6
MAJOR_MISSING_SCORE = -6
DETAIL_MATCH_SCORE = 6
DETAIL_MISSING_SCORE = -8
DETAIL_PREFIX_MATCH_SCORE = 3
DETAIL_PREFIX_CONFLICT_SCORE = -5
FIRST_SYMBOL_SCORE = 6
MISSING_SYMBOL_SCORE = -6
ADDITIONAL_SYMBOL_SCORE = 2
SUFFIX_SCORE = 2
DISCARDED_CHARACTER_SCORE = -1
NUMERIC_SYMBOL_RUN_SCORE = -6
MAX_RECONSTRUCTION_CHARACTERS = 256


@dataclass(frozen=True)
class ReconstructionResult:
    text: str
    major: str
    detail: str
    symbols: str
    suffix: str
    structure_score: int
    retained_positions: tuple[int, ...]
    discarded_character_count: int
    completed_section_count: int


@dataclass(frozen=True)
class _ParsePath:
    major: str = ""
    detail: str = ""
    symbols: str = ""
    suffix: str = ""
    score: int = 0
    retained_positions: tuple[int, ...] = ()
    discarded_character_count: int = 0
    completed_section_count: int = 0


def reconstruct_call_number(token_texts: Sequence[object]) -> ReconstructionResult:
    raw_parts = [" ".join(str(text).split()) for text in token_texts if str(text).strip()]
    characters = tuple(character for part in raw_parts for character in part if not character.isspace())
    length = len(characters)
    if length > MAX_RECONSTRUCTION_CHARACTERS:
        missing_score = MAJOR_MISSING_SCORE + DETAIL_MISSING_SCORE + MISSING_SYMBOL_SCORE
        return ReconstructionResult(
            text=" ".join(raw_parts),
            major="",
            detail="",
            symbols="",
            suffix="",
            structure_score=missing_score + length * DISCARDED_CHARACTER_SCORE,
            retained_positions=(),
            discarded_character_count=length,
            completed_section_count=0,
        )

    @lru_cache(maxsize=None)
    def consecutive_digit_count(index: int) -> int:
        count = 0
        while index + count < length and characters[index + count].isdigit():
            count += 1
        return count

    def starts_overlong_symbol_run(index: int) -> bool:
        return (
            index < length
            and characters[index].isalpha()
            and consecutive_digit_count(index + 1) > 3
        )

    @lru_cache(maxsize=None)
    def done(index: int) -> _ParsePath:
        if index >= length:
            return _ParsePath()
        return _discard(done(index + 1))

    @lru_cache(maxsize=None)
    def suffix(index: int, progress: int) -> _ParsePath | None:
        options: list[_ParsePath | None] = []
        if progress == 0:
            options.append(done(index))
        elif progress == 3 and index >= length:
            options.append(_add_score(_ParsePath(), SUFFIX_SCORE))
        if index >= length:
            return _best(options)

        character = characters[index]
        if progress != 3:
            options.append(_discard(suffix(index + 1, progress)))
        if progress == 0 and character.lower() in {"v", "c"}:
            options.append(_prepend(suffix(index + 1, 1), "suffix", character.lower(), index))
        elif progress == 1 and character == ".":
            options.append(_prepend(suffix(index + 1, 2), "suffix", character, index))
        elif progress == 2 and character.isdigit():
            options.append(_prepend(suffix(index + 1, 3), "suffix", character, index))
        elif progress == 3 and character.isdigit():
            options.append(_prepend(suffix(index + 1, 3), "suffix", character, index))
        return _best(options)

    def close_symbol_group(path: _ParsePath | None, has_completed_group: bool) -> _ParsePath | None:
        if has_completed_group:
            return _add_score(path, ADDITIONAL_SYMBOL_SCORE)
        return _add_score(path, FIRST_SYMBOL_SCORE, completed_sections=1)

    @lru_cache(maxsize=None)
    def symbols(
        index: int,
        mode: str,
        digit_count: int,
        has_completed_group: bool,
        numeric_run_active: bool,
    ) -> _ParsePath | None:
        options: list[_ParsePath | None] = []
        if mode == "start":
            options.append(_add_score(suffix(index, 0), MISSING_SYMBOL_SCORE))
        else:
            trailing_numeric_penalty = (
                NUMERIC_SYMBOL_RUN_SCORE
                if index < length and characters[index].isdigit()
                else 0
            )
            options.append(
                close_symbol_group(
                    _add_score(suffix(index, 0), trailing_numeric_penalty),
                    has_completed_group,
                )
            )
        if index >= length:
            return _best(options)

        character = characters[index]
        invalid_numeric = character.isdigit() and (
            mode == "start"
            or (mode == "letters" and consecutive_digit_count(index) > 3)
            or (
                mode == "digits"
                and digit_count + consecutive_digit_count(index) > 3
            )
        )
        starts_overlong_run = starts_overlong_symbol_run(index)
        numeric_penalty = (
            NUMERIC_SYMBOL_RUN_SCORE
            if (invalid_numeric or starts_overlong_run) and not numeric_run_active
            else 0
        )
        options.append(
            _discard(
                symbols(
                    index + 1,
                    mode,
                    digit_count,
                    has_completed_group,
                    invalid_numeric or starts_overlong_run,
                ),
                extra_score=numeric_penalty,
            )
        )

        if mode == "start" and _is_symbol_letter(character):
            options.append(
                _prepend(symbols(index + 1, "letters", 0, False, False), "symbols", character, index)
            )
        elif mode == "letters":
            if _is_symbol_letter(character):
                options.append(
                    _prepend(
                        symbols(index + 1, "letters", 0, has_completed_group, False),
                        "symbols",
                        character,
                        index,
                    )
                )
            elif character.isdigit():
                options.append(
                    _prepend(
                        symbols(index + 1, "digits", 1, has_completed_group, False),
                        "symbols",
                        character,
                        index,
                    )
                )
        elif mode == "digits":
            if character.isdigit() and digit_count < 3:
                options.append(
                    _prepend(
                        symbols(index + 1, "digits", digit_count + 1, has_completed_group, False),
                        "symbols",
                        character,
                        index,
                    )
                )
            elif _is_symbol_letter(character):
                continued = close_symbol_group(
                    symbols(index + 1, "letters", 0, True, False),
                    has_completed_group,
                )
                options.append(_prepend(continued, "symbols", character, index))
        return _best(options)

    def detail_score(major_digit: str, detail_digit: str) -> int:
        if not major_digit:
            return DETAIL_MATCH_SCORE
        relation = (
            DETAIL_PREFIX_MATCH_SCORE
            if major_digit == detail_digit
            else DETAIL_PREFIX_CONFLICT_SCORE
        )
        return DETAIL_MATCH_SCORE + relation

    @lru_cache(maxsize=None)
    def detail(
        index: int,
        progress: int,
        major_digit: str,
        detail_digit: str,
    ) -> _ParsePath | None:
        options: list[_ParsePath | None] = []
        if progress == 0:
            options.append(_add_score(symbols(index, "start", 0, False, False), DETAIL_MISSING_SCORE))
        elif progress in {3, 5}:
            options.append(
                _add_score(
                    symbols(index, "start", 0, False, False),
                    detail_score(major_digit, detail_digit),
                    completed_sections=1,
                )
            )
        if index >= length:
            return _best(options)

        character = characters[index]
        numeric_evasion_penalty = (
            NUMERIC_SYMBOL_RUN_SCORE
            if progress in {3, 4, 5} and starts_overlong_symbol_run(index)
            else 0
        )
        options.append(
            _discard(
                detail(index + 1, progress, major_digit, detail_digit),
                extra_score=numeric_evasion_penalty,
            )
        )
        if progress < 3 and character.isdigit():
            first_digit = character if progress == 0 else detail_digit
            options.append(
                _prepend(
                    detail(index + 1, progress + 1, major_digit, first_digit),
                    "detail",
                    character,
                    index,
                )
            )
        elif progress == 3 and character == ".":
            options.append(
                _prepend(detail(index + 1, 4, major_digit, detail_digit), "detail", character, index)
            )
        elif progress == 4 and character.isdigit():
            options.append(
                _prepend(detail(index + 1, 5, major_digit, detail_digit), "detail", character, index)
            )
        elif progress == 5 and character.isdigit():
            options.append(
                _prepend(detail(index + 1, 5, major_digit, detail_digit), "detail", character, index)
            )
        return _best(options)

    @lru_cache(maxsize=None)
    def major(index: int, progress: int, major_digit: str) -> _ParsePath | None:
        options: list[_ParsePath | None] = []
        if progress == 0:
            options.append(_add_score(detail(index, 0, "", ""), MAJOR_MISSING_SCORE))
        elif progress == 3:
            options.append(
                _add_score(
                    detail(index, 0, major_digit, ""),
                    MAJOR_MATCH_SCORE,
                    completed_sections=1,
                )
            )
        if index >= length:
            return _best(options)

        character = characters[index]
        options.append(_discard(major(index + 1, progress, major_digit)))
        if progress == 0 and character.isdigit():
            options.append(_prepend(major(index + 1, 1, character), "major", character, index))
        elif progress in {1, 2} and character == "0":
            options.append(_prepend(major(index + 1, progress + 1, major_digit), "major", character, index))
        return _best(options)

    selected = major(0, 0, "") or _ParsePath(
        score=MAJOR_MISSING_SCORE + DETAIL_MISSING_SCORE + MISSING_SYMBOL_SCORE
    )
    text = _format_sections(selected)
    if selected.completed_section_count == 0:
        text = " ".join(raw_parts)
    return ReconstructionResult(
        text=text,
        major=selected.major,
        detail=selected.detail,
        symbols=selected.symbols,
        suffix=selected.suffix,
        structure_score=selected.score,
        retained_positions=selected.retained_positions,
        discarded_character_count=selected.discarded_character_count,
        completed_section_count=selected.completed_section_count,
    )


def _format_sections(path: _ParsePath) -> str:
    return " ".join(
        section
        for section in (path.major, path.detail, path.symbols, path.suffix)
        if section
    )


def _is_symbol_letter(character: str) -> bool:
    return character.isalpha() and character not in {"V", "C"}


def _path_rank(path: _ParsePath) -> tuple[object, ...]:
    return (
        path.score,
        path.completed_section_count,
        -path.discarded_character_count,
        tuple(-position for position in path.retained_positions),
    )


def _best(paths: Sequence[_ParsePath | None]) -> _ParsePath | None:
    available = [path for path in paths if path is not None]
    return max(available, key=_path_rank) if available else None


def _add_score(
    path: _ParsePath | None,
    score: int,
    completed_sections: int = 0,
) -> _ParsePath | None:
    if path is None:
        return None
    return replace(
        path,
        score=path.score + score,
        completed_section_count=path.completed_section_count + completed_sections,
    )


def _discard(path: _ParsePath | None, extra_score: int = 0) -> _ParsePath | None:
    if path is None:
        return None
    return replace(
        path,
        score=path.score + DISCARDED_CHARACTER_SCORE + extra_score,
        discarded_character_count=path.discarded_character_count + 1,
    )


def _prepend(
    path: _ParsePath | None,
    section: str,
    character: str,
    position: int,
) -> _ParsePath | None:
    if path is None:
        return None
    return replace(
        path,
        **{
            section: character + getattr(path, section),
            "retained_positions": (position, *path.retained_positions),
        },
    )
