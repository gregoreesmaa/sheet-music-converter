"""Unit tests for build_unlabeled_10k.py: leakage rejection, NC rejection, determinism, schema."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_unlabeled_10k as b10k


def _write_png(path: Path, size=(200, 160), color=(240, 240, 240)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _write_jpg(path: Path, size=(200, 160), color=(230, 225, 215)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=90)
    return path


class ExclusionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = b10k.ExclusionPolicy.from_repo(b10k.REPO_ROOT)

    def test_rejects_gate20_ids_scores_and_upstream_lc(self):
        # Gate20 do_not_train page ID, parent score, held_out_4page_gate, and upstream lc ID
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\big500-renders\ds0072-3.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\big500-renders\ds0072-1.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\sq-renders\sq7070319-1.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\big500-renders\ds0005-2.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\s16sim-photos\lc5033089_mscz-1__sim.jpg")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\gate20\clean\anything.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\dev-12\foo.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\test-8\bar.png")))

        # Safe non-eval big500/sq/pdmx pages pass
        self.assertFalse(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\big500-renders\ds0001-1.png")))
        self.assertFalse(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\sq-renders\sq10307350-1.png")))
        self.assertFalse(self.policy.is_leaked_eval(Path(r"D:\tmp\corpus10x\pdmx\quota-renders\Qma12KU4Fg-1.png")))

    def test_rejects_suite31_slugs_without_false_positive_on_non_eval_hw(self):
        self.assertTrue(self.policy.is_leaked_eval(Path(r"data-gen\score-corpus\suite31\clean\repeats-melody.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\other\repeats-melody.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\other\mozart-k458-1.png")))
        self.assertTrue(self.policy.is_leaked_eval(Path(r"D:\tmp\other\beethoven-grosse-fuge.png")))
        # Non-eval handwritten Mozart K.458 and Beethoven Op.133 MUST be allowed
        self.assertFalse(
            self.policy.is_leaked_eval(
                Path(r"data-gen\score-corpus\handwritten-500\handwritten\mozart-k458\p001.png")
            )
        )
        self.assertFalse(
            self.policy.is_leaked_eval(
                Path(r"data-gen\score-corpus\handwritten-500\handwritten\beethoven-op133-feduol\p004.png")
            )
        )

    def test_rejects_all_5_held_out_suite31_hw_works(self):
        for work in (
            "beethoven-op135-ms",
            "beethoven-op135-sbb",
            "mozart-k464",
            "mozart-k387",
            "mozart-k421",
            "schubert-d810",
        ):
            p = Path(rf"data-gen\score-corpus\handwritten-500\handwritten\{work}\p001.png")
            self.assertTrue(self.policy.is_leaked_eval(p), f"Should reject {p}")

        for safe_work in (
            "beethoven-op127",
            "beethoven-op130",
            "beethoven-op131-sbb",
            "beethoven-op133-ms1",
            "mozart-k428",
            "mozart-k458",
            "mozart-k465",
            "mozart-k590",
        ):
            p = Path(rf"data-gen\score-corpus\handwritten-500\handwritten\{safe_work}\p001.png")
            self.assertFalse(self.policy.is_leaked_eval(p), f"Should allow {p}")

    def test_rejects_nc_and_muscima_paths(self):
        self.assertTrue(self.policy.is_forbidden_nc(Path(r"D:\tmp\corpus10x-NC\cvc-muscima\w-01\p001.png")))
        self.assertTrue(self.policy.is_forbidden_nc(Path(r"D:\tmp\cvc-muscima\foo.png")))
        self.assertTrue(self.policy.is_forbidden_nc(Path(r"D:\tmp\dcml\bar.png")))
        self.assertFalse(self.policy.is_forbidden_nc(Path(r"D:\tmp\corpus10x\pdmx\quota-renders\Qm123-1.png")))


class BuildCorpusEndToEndTest(unittest.TestCase):
    def _build_fixture_tree(self, root: Path):
        repo = root / "repo"
        data = root / "data"

        # Minimal gate20 + suite31 metadata
        g20 = repo / "data-gen" / "score-corpus" / "gate20"
        g20.mkdir(parents=True)
        (g20 / "exclusion.json").write_text(
            json.dumps({"do_not_train": ["ds0072-3"], "held_out_4page_gate": ["ds0005-1"]}),
            encoding="utf-8",
        )
        (g20 / "gate_manifest.json").write_text(
            json.dumps({"pages": [{"id": "ds0072-3", "score": "ds0072", "source": "lc5033089.mscx"}]}),
            encoding="utf-8",
        )
        s31 = repo / "data-gen" / "score-corpus" / "suite31"
        s31.mkdir(parents=True)
        (s31 / "suite31_manifest.json").write_text(
            json.dumps({"pieces": [{"slug": "repeats-melody"}]}),
            encoding="utf-8",
        )

        # Photo pool (+ 1 leaked gate20 lc + 1 corrupt)
        s16 = data / "s16sim-photos"
        _write_jpg(s16 / "lc5033089_sim.jpg")  # MUST be rejected
        for i in range(4):
            _write_jpg(s16 / f"ok_photo_{i}.jpg")
        (s16 / "corrupt_photo.jpg").write_bytes(b"not-a-jpeg")

        qsys = data / "corpus10x" / "pdmx" / "quota-sysdata"
        for i in range(6):
            _write_jpg(qsys / f"QmPhoto{i}-1-s0_d0.jpg")
            _write_png(qsys / f"QmHand{i}-1-s0_h0.png")

        # Clean render pool (+ 1 leaked gate20 + 1 leaked suite31)
        b500 = repo / "data-gen" / "score-corpus" / "big500-renders"
        _write_png(b500 / "ds0072-3.png")  # MUST be rejected
        _write_png(b500 / "repeats-melody.png")  # MUST be rejected
        for i in range(6):
            _write_png(b500 / f"ds0001-{i + 1}.png")

        # Forbidden NC directory (must never be touched/included)
        _write_png(data / "corpus10x-NC" / "cvc-muscima" / "bad.png")

        # Real handwritten (+ 2 held-out eval works that MUST be rejected)
        hw = repo / "data-gen" / "score-corpus" / "handwritten-500" / "handwritten"
        _write_png(hw / "beethoven-op135-sbb" / "p001.png", size=(400, 600))
        _write_png(hw / "mozart-k464" / "p001.png", size=(400, 600))
        _write_png(hw / "beethoven-op127" / "p001.png", size=(400, 600))
        _write_png(hw / "mozart-k428" / "p001.png", size=(400, 600))

        return repo, data

    def test_deterministic_seeded_build_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, data = self._build_fixture_tree(root)
            quotas = {
                "photo": 6,
                "clean_render": 4,
                "hand_ink_scan": 4,
                "real_handwritten": 5,  # 2 pages + 3 band crops
            }

            out1 = root / "out1"
            out2 = root / "out2"
            s1 = b10k.build_corpus(out_dir=out1, repo_root=repo, data_root=data, seed=42, domain_quotas=quotas)
            s2 = b10k.build_corpus(out_dir=out2, repo_root=repo, data_root=data, seed=42, domain_quotas=quotas)

            self.assertEqual(s1["total"], 19)
            self.assertEqual(s1["domain_counts"], quotas)
            self.assertEqual(s1["leaked_eval"], 0)
            self.assertEqual(s1["nc_count"], 0)
            self.assertEqual(s1["corrupt_count"], 0)
            self.assertEqual(s1["label_key_violations"], 0)
            self.assertGreaterEqual(s1["rejected_eval_candidates"], 5)

            # Determinism: same IDs, domains, source_pools, and basenames across runs
            rows1 = [json.loads(l) for l in (out1 / "unlabeled_10k_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            rows2 = [json.loads(l) for l in (out2 / "unlabeled_10k_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows1), 19)
            sig1 = [(r["id"], r["domain"], r["granularity"], r["source_pool"], Path(r["path"]).name) for r in rows1]
            sig2 = [(r["id"], r["domain"], r["granularity"], r["source_pool"], Path(r["path"]).name) for r in rows2]
            self.assertEqual(sig1, sig2)

            # Schema check on every row: exact required keys and zero label keys
            for r in rows1:
                self.assertEqual(set(r.keys()), b10k.REQUIRED_ROW_KEYS)
                self.assertFalse(set(r.keys()) & b10k.FORBIDDEN_LABEL_KEYS)
                self.assertGreater(r["width"], 0)
                self.assertGreater(r["height"], 0)
                self.assertTrue(Path(r["path"]).is_file())

    def test_audit_manifest_catches_leaks_nc_and_labels(self):
        policy = b10k.ExclusionPolicy.from_repo(b10k.REPO_ROOT)
        bad_rows = [
            {
                "id": "u10k-00000",
                "path": r"D:\tmp\corpus10x-NC\cvc-muscima\p1.png",
                "domain": "real_handwritten",
                "granularity": "page",
                "width": 100,
                "height": 100,
                "source_pool": "bad",
                "license": "NC",
                "label": "clef-G2",
            },
            {
                "id": "u10k-00001",
                "path": r"data-gen\score-corpus\handwritten-500\handwritten\schubert-d810\p002.png",
                "domain": "real_handwritten",
                "granularity": "page",
                "width": 100,
                "height": 100,
                "source_pool": "bad",
                "license": "PD",
            },
        ]
        audit = b10k.audit_manifest(bad_rows, policy, verify_headers=False)
        self.assertEqual(audit["nc_count"], 1)
        self.assertEqual(audit["leaked_eval"], 1)
        self.assertEqual(audit["label_key_violations"], 1)


if __name__ == "__main__":
    unittest.main()
