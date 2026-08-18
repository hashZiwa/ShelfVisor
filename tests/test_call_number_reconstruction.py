import unittest

from app.analysis.ocr.call_number_reconstruction import reconstruct_call_number


class CallNumberReconstructionTests(unittest.TestCase):
    def test_token_splits_do_not_change_reconstruction(self):
        combined = reconstruct_call_number(["500", "519.5", "ㅅ", "21", "c.2"])
        split = reconstruct_call_number(["5", "00", "519", ".5", "ㅅ21", "c", ".2"])

        self.assertEqual(combined.text, "500 519.5 ㅅ21 c.2")
        self.assertEqual(split.text, combined.text)
        self.assertEqual(split.structure_score, combined.structure_score)

    def test_integer_detail_is_valid(self):
        result = reconstruct_call_number(["500", "519", "ㅅ21"])

        self.assertEqual(result.text, "500 519 ㅅ21")

    def test_zero_major_preserves_detail_leading_zero(self):
        result = reconstruct_call_number(["000", "005.12", "ㄱ7"])

        self.assertEqual(result.major, "000")
        self.assertEqual(result.detail, "005.12")
        self.assertEqual(result.text, "000 005.12 ㄱ7")

    def test_major_detail_match_scores_above_conflict(self):
        matching = reconstruct_call_number(["500", "519", "ㅅ21"])
        conflicting = reconstruct_call_number(["500", "619", "ㅅ21"])

        self.assertEqual(matching.structure_score - conflicting.structure_score, 8)

    def test_noise_is_deleted_between_and_inside_tokens(self):
        between = reconstruct_call_number(["500", "ㄹ", "519.5", "ㅅ21", "c.2"])
        inside = reconstruct_call_number(["500", "ㄹ519.5", "ㅅ21", "c.2"])

        self.assertEqual(between.text, "500 519.5 ㅅ21 c.2")
        self.assertEqual(inside.text, between.text)
        self.assertEqual(inside.discarded_character_count, 1)

    def test_repeated_symbol_groups_share_one_output_section(self):
        result = reconstruct_call_number(["500", "519.5", "ㅅ", "21", "한"])

        self.assertEqual(result.symbols, "ㅅ21한")
        self.assertEqual(result.text, "500 519.5 ㅅ21한")

    def test_split_and_uppercase_suffixes_are_normalized(self):
        variants = (["V.2"], ["v", ".2"], ["v.", "2"])

        for suffix_tokens in variants:
            with self.subTest(suffix_tokens=suffix_tokens):
                result = reconstruct_call_number(["500", "519.5", "ㅅ21", *suffix_tokens])
                self.assertEqual(result.suffix, "v.2")

    def test_missing_suffix_period_is_not_invented(self):
        result = reconstruct_call_number(["500", "519.5", "ㅅ21", "v2"])

        self.assertNotEqual(result.suffix, "v.2")

    def test_numeric_leading_symbol_run_receives_large_penalty(self):
        valid = reconstruct_call_number(["500", "519.5", "ㅅ21", "한"])
        numeric_leading = reconstruct_call_number(["500", "519.5", "ㅅ21", "123", "한"])

        self.assertGreaterEqual(valid.structure_score - numeric_leading.structure_score, 6)

    def test_unstructured_input_uses_raw_fallback(self):
        result = reconstruct_call_number(["@@", "--", "??"])

        self.assertEqual(result.text, "@@ -- ??")
        self.assertEqual(result.completed_section_count, 0)


if __name__ == "__main__":
    unittest.main()
