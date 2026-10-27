import numpy as np

from vibe_voice.vad import UtteranceSegmenter


def _segmenter() -> UtteranceSegmenter:
    # 1 sample == 1 ms, so the thresholds below are easy to count.
    return UtteranceSegmenter(
        sample_rate=1000,
        min_speech_ms=100,
        min_silence_ms=50,
        max_speech_ms=500,
        speech_pad_ms=10,
    )


def test_silence_after_speech_closes_one_utterance():
    segmenter = _segmenter()
    assert segmenter.push(np.zeros(20), False) == []
    assert segmenter.push(np.ones(100), True) == []
    emitted = segmenter.push(np.zeros(50), False)
    assert len(emitted) == 1
    # 10 ms preroll + 100 ms speech + 10 ms end pad.
    assert emitted[0].shape == (120,)
    assert emitted[0][0] == 0
    assert emitted[0][10] == 1


def test_short_burst_is_dropped():
    segmenter = _segmenter()
    assert segmenter.push(np.ones(40), True) == []
    assert segmenter.push(np.zeros(50), False) == []
    assert segmenter.flush() == []


def test_short_pause_stays_inside_the_utterance():
    segmenter = _segmenter()
    segmenter.push(np.ones(100), True)
    assert segmenter.push(np.zeros(20), False) == []
    segmenter.push(np.full(30, 0.5), True)
    emitted = segmenter.push(np.zeros(50), False)
    assert len(emitted) == 1
    assert emitted[0].shape == (160,)  # 100 + 20 gap + 30 + 10 pad


def test_max_speech_cuts_without_waiting_for_silence():
    segmenter = _segmenter()
    assert segmenter.push(np.ones(400), True) == []
    emitted = segmenter.push(np.ones(150), True)
    assert len(emitted) == 1
    assert emitted[0].shape == (550,)


def test_flush_emits_open_speech_and_two_utterances_in_a_row():
    segmenter = _segmenter()
    segmenter.push(np.ones(120), True)
    flushed = segmenter.flush()
    assert len(flushed) == 1
    assert flushed[0].shape == (120,)

    segmenter.push(np.ones(100), True)
    first = segmenter.push(np.zeros(50), False)
    segmenter.push(np.ones(100), True)
    second = segmenter.push(np.zeros(50), False)
    assert len(first) == 1
    assert len(second) == 1
