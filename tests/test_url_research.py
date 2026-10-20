import unittest

from src.url_research import URL_FEATURE_COLUMNS, extract_url_features, load_phishvn_splits


class TestUrlResearch(unittest.TestCase):
    def test_feature_extraction_is_scheme_independent(self):
        plain = extract_url_features("tiktokshopseller-vn.com")
        explicit = extract_url_features("http://tiktokshopseller-vn.com")
        self.assertEqual(plain, explicit)
        self.assertEqual(plain["url_len"], 23.0)
        self.assertEqual(plain["letter_cnt"], 21.0)
        self.assertEqual(plain["special_cnt"], 2.0)
        self.assertNotIn("is_https", plain)

    def test_verified_phishvn_splits_match_feature_contract(self):
        splits = load_phishvn_splits()
        self.assertEqual(set(splits), {"train", "val", "test"})
        for frame in splits.values():
            self.assertTrue(frame["tier"].isin(["gold", "silver"]).all())
            self.assertEqual(set(frame["label"].unique()), {0, 1})
            self.assertTrue(set(URL_FEATURE_COLUMNS).issubset(frame.columns))
            self.assertIn("is_https", frame.columns)


if __name__ == "__main__":
    unittest.main()
