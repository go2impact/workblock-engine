from workblock_engine.utils.keystroke_compression import compress_keystrokes


class TestCompressKeystrokes:
    def test_empty_string(self):
        assert compress_keystrokes("") == ""

    def test_no_repeats(self):
        assert compress_keystrokes("hello") == "hello"

    def test_repeated_plain_text_over_threshold(self):
        assert compress_keystrokes("aaaaa") == "a{5}"

    def test_whitespace_runs_are_preserved(self):
        assert compress_keystrokes("a   a") == "a   a"

    def test_repeated_keystroke_token(self):
        assert compress_keystrokes("<Cmd><Cmd><Cmd>") == "<Cmd>{3}"

    def test_repeated_multi_token_group(self):
        result = compress_keystrokes("<Ctrl><Ctrl + a><Ctrl + c><Ctrl><Ctrl + a><Ctrl + c>")
        assert result == "(<Ctrl><Ctrl + a><Ctrl + c>){2}"

    def test_mixed_tokens_and_text(self):
        result = compress_keystrokes("<Backspace><Backspace><Backspace>hello")
        assert result == "<Backspace>{3}hello"

    def test_none_returns_none(self):
        assert compress_keystrokes(None) is None
