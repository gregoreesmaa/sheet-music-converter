"""Self-contained checks for tmp-train/mvd/data.py.

Builds tiny fixtures (PNGs, bekern txt, COCO, manifests) in a temp dir:
no repo data, no network, no torch. Run with the repo-root .venv python:

    .venv/Scripts/python.exe tmp-train/mvd/test_data.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402


def _write_png(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((9, 12, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _write_page(directory, stem, tokens):
    png = directory / (stem + ".png")
    _write_png(png, abs(hash(stem)) % (2 ** 32))
    (directory / (stem + "-bekern.txt")).write_text(" ".join(tokens) + "\n",
                                                   encoding="utf-8")
    return png


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")


def _write_coco(path, stems):
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(data.AUX_CLASSES)]
    images, anns = [], []
    aid = 1
    for iid, stem in enumerate(stems, 1):
        images.append({"id": iid, "file_name": stem + ".png",
                       "width": 12, "height": 9, "page": stem})
        anns.append({"id": aid, "image_id": iid, "category_id": 1,
                     "bbox": [1.0, 2.0, 3.0, 4.0], "area": 12.0, "iscrowd": 0})
        aid += 1
        anns.append({"id": aid, "image_id": iid, "category_id": 10,
                     "bbox": [0.0, 0.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0})
        aid += 1
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(path, "w", encoding="utf-8"))


TOKENS_A = ["clef-G-2", "key+0", "time-4-4", "quarter", "C", "4",
            "quarter", "D", "4", "bar"]
TOKENS_B = ["clef-F-4", "key+2", "time-3-4", "half", "rest", "dot",
            "eighth", "E", "5", "a1", "tie-start", "eighth", "E", "5",
            "tie-stop", "bar"]


def test_vocab_roundtrip():
    assert len(data.VOCAB) == len(set(data.VOCAB)), "VOCAB has duplicates"
    for fam in (["A", "G"], ["0", "8"], ["whole", "64th"],
                ["dot", "tie-start", "tie-stop"], ["a-2", "a2"],
                ["rest", "bar"],
                ["clef-G-2", "clef-F-4", "clef-C-3"], ["key-7", "key+0", "key+7"],
                ["time-4-4", "time-12-8", "time-4-8", "time-7-8",
                 "time-8-8"],
                ["tuplet-start-3-2", "tuplet-start-6-4", "tuplet-stop"],
                ["bar-rest"], ["breve"], ["chord"],
                ["voice-1", "voice-8", "voice-9", "voice-12"]):
        for tok in fam:
            assert tok in data.TOKEN_TO_ID, tok
    assert data.decode(data.encode(data.VOCAB)) == list(data.VOCAB)
    assert data.decode(data.encode("quarter C 4")) == ["quarter", "C", "4"]
    for bad_call in (lambda: data.encode(["nope"]),
                     lambda: data.encode("quarter nope"),
                     lambda: data.decode([len(data.VOCAB)]),
                     lambda: data.decode([-1])):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError from %r" % (bad_call,))
    print("PASS vocab: %d tokens, full round-trip identity" % len(data.VOCAB))


def test_piece_key():
    cases = {"ds0001/ds0001-1": "ds0001", "ds0001-1.aug0": "ds0001",
             "ds0001-1_d0": "ds0001", "sample01-1_degraded": "sample01",
             "song-1.aug3": "song", "plain": "plain"}
    for given, want in cases.items():
        assert data.piece_key(given) == want, (given, data.piece_key(given))
    print("PASS piece_key: %d cases" % len(cases))


def test_load_item(tmp):
    png = _write_page(tmp, "pageX-1", TOKENS_A)
    item = data.load_item(str(png), str(tmp / "pageX-1-bekern.txt"))
    assert set(item) == {"image", "tokens", "page_id"}, set(item)
    arr = item["image"]
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32, arr.dtype
    assert arr.shape == (9, 12, 3) and arr.min() >= 0.0 and arr.max() <= 1.0
    assert item["tokens"] == TOKENS_A and item["page_id"] == "pageX-1"
    print("PASS load_item: float32 HWC [0,1], tokens, page_id")


def _clean_tree(tmp):
    _write_page(tmp, "expA-1", TOKENS_A)
    _write_page(tmp, "expA-2", TOKENS_A)
    _write_page(tmp, "expA-1_d0", TOKENS_B)
    _write_page(tmp, "expB-1", TOKENS_B)
    man = tmp / "manifest.jsonl"
    _write_manifest(man, [
        {"id": "expA-1", "split": "train", "image": "expA-1.png"},
        {"id": "expA-2", "split": "train", "image": "expA-2.png"},
        {"id": "expA-1_d0", "split": "train", "image": "expA-1_d0.png"},
        {"id": "expB-1", "split": "val", "image": "expB-1.png"},
    ])
    coco = tmp / "coco.json"
    _write_coco(coco, ["expA-1", "expA-2", "expA-1_d0", "expB-1"])
    return man, coco


def test_load_split_clean(tmp):
    man, coco = _clean_tree(tmp)
    items = list(data.load_split(str(man), str(coco), "train"))
    assert [i["page_id"] for i in items] == ["expA-1", "expA-2", "expA-1_d0"]
    assert items[0]["tokens"] == TOKENS_A
    assert items[1]["tokens"] == TOKENS_A
    assert items[2]["tokens"] == TOKENS_B
    for item in items:
        assert set(item) == {"image", "tokens", "boxes", "page_id"}
        assert item["image"].dtype == np.float32 and item["image"].ndim == 3
        assert len(item["boxes"]) == 2, item["boxes"]
    b0, b1 = items[0]["boxes"]
    assert b0["bbox"] == [1.0, 2.0, 4.0, 6.0] and b0["category"] == "NoteBlack"
    assert b1["bbox"] == [0.0, 0.0, 2.0, 2.0] and b1["category"] == "Flag"
    val = list(data.load_split(str(man), str(coco), "val"))
    assert [i["page_id"] for i in val] == ["expB-1"]
    assert list(data.load_split(str(man), str(coco), "test")) == []
    try:
        data.load_split(str(man), str(coco), "heldout")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad split name")
    print("PASS load_split clean: 3 train + 1 val, boxes xyxy, empty test")


def test_load_split_straddled(tmp):
    _write_page(tmp, "song-1", TOKENS_A)
    _write_page(tmp, "song-2", TOKENS_A)
    man = tmp / "manifest.jsonl"
    _write_manifest(man, [
        {"id": "song-1", "split": "train", "image": "song-1.png"},
        {"id": "song-2", "split": "val", "image": "song-2.png"},
    ])
    coco = tmp / "coco.json"
    _write_coco(coco, ["song-1", "song-2"])
    try:
        data.load_split(str(man), str(coco), "train")
    except AssertionError as exc:
        assert "song" in str(exc), str(exc)
        print("PASS load_split straddled page-suffix raises: %s" % exc)
    else:
        raise AssertionError("straddled manifest did NOT raise")

    _write_page(tmp, "song-1_d0", TOKENS_B)
    _write_page(tmp, "song-1.aug1", TOKENS_B)
    _write_manifest(man, [
        {"id": "song-1", "split": "train", "image": "song-1.png"},
        {"id": "song-1_d0", "split": "val", "image": "song-1_d0.png"},
        {"id": "song-1.aug1", "split": "train", "image": "song-1.aug1.png"},
    ])
    try:
        data.load_split(str(man), str(coco), "train")
    except AssertionError as exc:
        print("PASS load_split straddled aug/degrade raises: %s" % exc)
    else:
        raise AssertionError("aug/degrade-straddled manifest did NOT raise")


class VocabPlusTest(unittest.TestCase):
    """T3.1 item 5: the bekern+ vocab extension is append-only.

    The 23 new tokens (grace, slur/articulation/fermata, dynamics,
    hairpins, barline variants, time-1-16/1-32) sit after the 133
    historical tokens, so no pre-existing id shifts; every new token
    encodes and decodes.
    """

    PRE = 133  # len(VOCAB) before the T3.1 bekern+ append
    POST_BEKERNPLUS = 156  # len(VOCAB) before the T3.2 t32-corpus append
    POST_T32 = 175  # len(VOCAB) before the leland 128th append
    POST_ITER7 = 233  # len(VOCAB) before bekern+v2 Phase A append

    def test_size(self):
        self.assertEqual(len(data._BEKERNPLUS), 23)
        self.assertEqual(len(data._T32_ADD), 19)
        self.assertEqual(len(data._LELAND_ADD), 1)
        self.assertEqual(len(data._BEKERN_V2A_ADD), 28)
        self.assertEqual(len(data._BASE_PRE_V2A), self.POST_ITER7)
        self.assertEqual(len(data.VOCAB), self.POST_ITER7 + 28)
        self.assertEqual(len(data.VOCAB), 261)

    def test_append_only_id_stability(self):
        self.assertEqual(
            list(data.VOCAB[self.POST_ITER7:]),
            list(data._BEKERN_V2A_ADD),
        )
        self.assertEqual(
            list(data.VOCAB[self.PRE:self.POST_BEKERNPLUS]),
            list(data._BEKERNPLUS),
        )
        self.assertEqual(
            list(data.VOCAB[self.POST_BEKERNPLUS:self.POST_T32]),
            list(data._T32_ADD),
        )
        self.assertEqual(
            list(data.VOCAB[self.POST_T32:self.POST_T32 + len(data._LELAND_ADD)]),
            list(data._LELAND_ADD),
        )
        self.assertEqual(len(set(data.VOCAB)), len(data.VOCAB))
        for old in ("quarter", "C", "4", "rest", "bar", "tie-start",
                    "clef-G-2", "key+0", "time-4-4", "voice-1",
                    "tuplet-start-3-2", "bar-rest", "breve", "chord"):
            self.assertLess(data.TOKEN_TO_ID[old], self.PRE, old)
        for v2tok in data._BEKERN_V2A_ADD:
            self.assertGreaterEqual(data.TOKEN_TO_ID[v2tok], self.POST_ITER7, v2tok)
            self.assertEqual(data.decode(data.encode([v2tok])), [v2tok])
        # build_vocab with include_ext_dynamics=True stays duplicate-free
        ext_vocab, _ = data.build_vocab(include_ext_dynamics=True)
        self.assertEqual(ext_vocab[:len(data.VOCAB)], list(data.VOCAB))
        self.assertEqual(len(ext_vocab), len(set(ext_vocab)))
        self.assertEqual(data.PAD_ID, len(data.VOCAB))
        self.assertEqual(data.BOS_ID, len(data.VOCAB) + 1)
        self.assertEqual(data.EOS_ID, len(data.VOCAB) + 2)

    def test_new_tokens_encode_decode(self):
        for tok in data._BEKERNPLUS:
            self.assertEqual(data.decode(data.encode([tok])), [tok])
        stream = ("clef-G-2 key+0 time-4-4 voice-1 quarter C 4 grace "
                  "eighth D 5 quarter E 4 slur-start stacc quarter F 4 "
                  "slur-stop tie-start bar dyn-f hairpin-cresc-start "
                  "hairpin-stop bar-light-heavy")
        self.assertEqual(data.decode(data.encode(stream)), stream.split())

    def test_t32_tokens_encode_decode(self):
        for tok in data._T32_ADD:
            self.assertEqual(data.decode(data.encode([tok])), [tok])
        self.assertEqual(len(data._T32_ADD), 19)
        self.assertIn("time-9-4", data.TOKEN_TO_ID)
        self.assertIn("tuplet-start-11-12", data.TOKEN_TO_ID)
        # prefix byte-identity: first 156 ids predate the T3.2 append
        for old in ("quarter", "C", "4", "rest", "bar", "tie-start",
                    "clef-G-2", "key+0", "time-4-4", "voice-1",
                    "tuplet-start-3-2", "bar-rest", "breve", "chord",
                    "grace", "slur-start", "time-1-16"):
            self.assertLess(data.TOKEN_TO_ID[old], self.POST_BEKERNPLUS, old)

    def test_leland_tokens_encode_decode(self):
        for tok in data._LELAND_ADD:
            self.assertEqual(data.decode(data.encode([tok])), [tok])
        self.assertEqual(len(data._LELAND_ADD), 1)
        self.assertIn("128th", data.TOKEN_TO_ID)
        # prefix byte-identity: first 175 ids predate the leland append
        for old in ("quarter", "C", "4", "rest", "bar", "tie-start",
                    "clef-G-2", "key+0", "time-4-4", "voice-1",
                    "tuplet-start-3-2", "bar-rest", "breve", "chord",
                    "grace", "slur-start", "time-1-16", "time-9-4",
                    "tuplet-start-11-12"):
            self.assertLess(data.TOKEN_TO_ID[old], self.POST_T32, old)

    def test_new_tokens_unknown_to_old_sequences(self):
        with self.assertRaises(ValueError):
            data.encode(["grace-x"])
        self.assertNotIn("grace-x", data.TOKEN_TO_ID)


def main():
    assert "torch" not in sys.modules, "data.py must not import torch"
    test_vocab_roundtrip()
    test_piece_key()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_load_item(tmp)
        test_load_split_clean(tmp)
    with tempfile.TemporaryDirectory() as td:
        test_load_split_straddled(Path(td))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
