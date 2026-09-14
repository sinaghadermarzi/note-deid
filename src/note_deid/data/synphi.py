"""Build the ``synphi`` benchmark: source notes -> cleaned -> PHI-injected -> stratified splits + manifest.

    python -m note_deid.data.synphi build --profile configs/synphi/i2b2like.yaml --out data/synphi/v0.1 \
        [--demo] [--mtsamples path/to/mtsamples.csv] [--asclepius 500] [--limit N] [--seed 0] [--version 0.1]
    python -m note_deid.data.synphi export-opf --bench data/synphi/v0.1 [--splits train dev test]

``export-opf`` writes ``<split>.opf.jsonl`` next to the unified files for the upstream ``opf train`` / ``opf eval``
CLI (labels mapped to OPF's eight; subtypes OPF cannot represent are dropped). ``note_deid.tc.train`` reads the
unified files directly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from note_deid.data.phi_injection.inject import Injector, Profile
from note_deid.labels import I2B2_TO_OPF, INSTITUTIONAL_SUBTYPES
from note_deid.schema import Doc, read_jsonl, to_opf_record, write_jsonl

SPLITS = ("train", "dev", "test")


def stratified_split(
    docs: Sequence[Doc], ratios: tuple[float, float, float] = (0.7, 0.1, 0.2), seed: int = 0, key: str = "note_type"
) -> dict[str, list[Doc]]:
    """Split by source document, stratified by ``meta[key]``, deterministic for a seed."""
    rng = random.Random(seed)
    groups: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        groups[str(d.meta.get(key, ""))].append(d)
    out: dict[str, list[Doc]] = {s: [] for s in SPLITS}
    carry = [0.0, 0.0]  # rounding remainders carried across strata so small groups still fill every split
    for _, members in sorted(groups.items()):
        members = sorted(members, key=lambda d: d.doc_id)
        rng.shuffle(members)
        n = len(members)
        want_train = ratios[0] * n + carry[0]
        n_train = min(n, max(0, int(round(want_train))))
        carry[0] = want_train - n_train
        want_dev = ratios[1] * n + carry[1]
        n_dev = min(n - n_train, max(0, int(round(want_dev))))
        carry[1] = want_dev - n_dev
        out["train"].extend(members[:n_train])
        out["dev"].extend(members[n_train : n_train + n_dev])
        out["test"].extend(members[n_train + n_dev :])
    return out


def label_counts(docs: Iterable[Doc]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for d in docs:
        for s in d.spans:
            c[s.label] += 1
    return dict(sorted(c.items()))


def build(
    sources: Sequence[Doc],
    profile: Profile,
    out_dir: str | Path,
    seed: int = 0,
    version: str = "0.1",
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    injector = Injector(profile, seed)
    injected = [injector.inject(d) for d in sources]
    splits = stratified_split(injected, ratios, seed)
    manifest: dict[str, Any] = {
        "name": "synphi",
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "profile": profile.to_dict(),
        "profile_fingerprint": profile.fingerprint(),
        "sources": dict(Counter(str(d.meta.get("source", "")) for d in sources)),
        "n_docs": len(injected),
        "splits": {},
        "source_ids": {s: [d.doc_id for d in docs] for s, docs in splits.items()},
    }
    for split, docs in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", docs)
        manifest["splits"][split] = {
            "n_docs": len(docs),
            "n_spans": sum(len(d.spans) for d in docs),
            "label_counts": label_counts(docs),
            "ambiguous_docs": sum(1 for d in docs if d.meta.get("ambiguous_strings")),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def export_opf(
    bench_dir: str | Path, splits: Sequence[str] = SPLITS, source: str = "synphi"
) -> dict[str, dict[str, int]]:
    """Write ``<split>.opf.jsonl`` for the upstream OPF CLI: canonical labels mapped through ``I2B2_TO_OPF``, spans
    of subtypes OPF cannot represent (and the institutional extension) dropped. Returns kept / dropped span counts
    per split."""
    bench_dir = Path(bench_dir)
    opf_map: dict[str, str | None] = {**I2B2_TO_OPF, **dict.fromkeys(INSTITUTIONAL_SUBTYPES, None)}
    report: dict[str, dict[str, int]] = {}
    for split in splits:
        src = bench_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"{src} does not exist; build the benchmark first")
        kept = dropped = 0
        with (bench_dir / f"{split}.opf.jsonl").open("w", encoding="utf-8") as fh:
            for doc in read_jsonl(src):
                rec = to_opf_record(doc, opf_map, source)
                n_kept = sum(len(v) for v in rec["spans"].values())
                kept += n_kept
                dropped += len(doc.spans) - n_kept
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report[split] = {"kept": kept, "dropped": dropped}
    return report


def _load_sources(ns: argparse.Namespace) -> list[Doc]:
    docs: list[Doc] = []
    if ns.demo:
        from note_deid.data.samples import demo_docs

        docs.extend(demo_docs())
    if ns.mtsamples:
        from note_deid.data.mtsamples import load_mtsamples

        docs.extend(load_mtsamples(ns.mtsamples))
    if ns.asclepius:
        from note_deid.data.asclepius import load_asclepius

        docs.extend(load_asclepius(limit=ns.asclepius))
    if ns.limit:
        docs = docs[: ns.limit]
    if not docs:
        raise SystemExit("no source documents: pass --demo, --mtsamples CSV and/or --asclepius N")
    return docs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a synphi version")
    b.add_argument("--profile", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--version", default="0.1")
    b.add_argument("--demo", action="store_true", help="use the built-in PHI-free demo notes")
    b.add_argument("--mtsamples", help="path to mtsamples.csv")
    b.add_argument("--asclepius", type=int, default=0, help="number of Asclepius notes to stream from the HF hub")
    b.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("export-opf", help="write <split>.opf.jsonl for the upstream `opf train` / `opf eval` CLI")
    e.add_argument("--bench", required=True, help="benchmark directory produced by `build`")
    e.add_argument("--splits", nargs="+", default=list(SPLITS))
    ns = ap.parse_args(argv)
    if ns.cmd == "export-opf":
        print(json.dumps(export_opf(ns.bench, ns.splits), indent=2))
        return
    profile = Profile.load(ns.profile)
    manifest = build(_load_sources(ns), profile, ns.out, ns.seed, ns.version)
    print(json.dumps({k: manifest[k] for k in ("version", "n_docs", "splits")}, indent=2))


if __name__ == "__main__":
    main()
