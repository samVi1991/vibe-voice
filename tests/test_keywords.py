import pytest

from vibe_voice.errors import KeywordFileError
from vibe_voice.keywords import load_keywords, packaged_keywords_path, parse_keywords


def test_default_keyword_file_is_nihao_xiaozhu():
    entries = load_keywords(packaged_keywords_path())
    assert len(entries) == 1
    keyword = entries[0]
    assert keyword.phrase == "你好小助"
    assert keyword.tokens == ("n", "ǐ", "h", "ǎo", "x", "iǎo", "zh", "ù")
    assert keyword.boost == 1.5
    assert keyword.threshold == 0.25
    assert keyword.to_sherpa_line() == "n ǐ h ǎo x iǎo zh ù :1.5 #0.25 @你好小助"


def test_comments_blank_lines_and_second_keyword():
    text = """
    # 这是注释
    // 也是注释

    n ǐ h ǎo x iǎo zh ù @你好小助
    x iǎo ài t óng x ué :2.0 #0.6 @小爱同学
    """
    entries = parse_keywords(text)
    assert [item.phrase for item in entries] == ["你好小助", "小爱同学"]
    assert entries[1].tokens == ("x", "iǎo", "ài", "t", "óng", "x", "ué")
    assert entries[1].boost == 2.0
    assert entries[1].threshold == 0.6


def test_rejects_bad_score_and_empty_phrase():
    with pytest.raises(KeywordFileError):
        parse_keywords("n ǐ : 1.5 @你好")
    with pytest.raises(KeywordFileError):
        parse_keywords("n ǐ @")
    with pytest.raises(KeywordFileError):
        parse_keywords("# 只有注释\n\n")
    with pytest.raises(KeywordFileError):
        parse_keywords("n ǐ h ǎo :1.5 多余 @你好")
