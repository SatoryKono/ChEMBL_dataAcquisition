"""Protein classification utilities.

This module provides deterministic classification of proteins into hierarchical
classes (L1/L2/L3) based on high-confidence signals extracted from UniProt JSON
entries. The rules implement the specification described in the project
instructions and rely solely on information present in the JSON object.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationResult:
    """Container for an intermediate classification decision."""

    label: str
    rule_id: str
    evidence: List[str]
    confidence: str


SignalDict = Dict[str, Set[str]]

# ---------------------------------------------------------------------------
# Constants for GO IDs and hint patterns
# ---------------------------------------------------------------------------

GO_TF = "GO:0003700"
GO_GPCR = "GO:0004930"
GO_ION_CHANNEL = "GO:0005216"
GO_TRANSPORTER = "GO:0005215"
GO_NUCLEAR_R = "GO:0004879"
GO_CATALYTIC = "GO:0003824"
GO_KINASE = "GO:0004672"
GO_PROTEASE = "GO:0008233"

GO_PHOSPHATASE_IDS = {
    "GO:0004721",  # phosphoprotein phosphatase activity
    "GO:0004722",  # protein serine/threonine phosphatase activity
    "GO:0004725",  # protein tyrosine phosphatase activity
}

HINTS: Dict[str, List[str]] = {
    "TF": ["transcription factor", "homeobox", "bhlh", "bzip", "zinc-finger"],
    "GPCR": [
        "g-protein coupled receptor",
        "g protein-coupled receptor",
        "7tm",
        "seven-transmembrane",
    ],
    "ION_CHANNEL": ["ion channel"],
    "TRANSPORTER": ["transport", "transporter", "abc transporter", "slc family"],
    "NUCLEAR_R": ["nuclear receptor"],
    "KINASE": [
        "protein kinase",
        "serine/threonine-protein kinase",
        "tyrosine-protein kinase",
        "rtk",
    ],
    "PROTEASE": ["protease", "peptidase", "merops"],
    "PHOSPHATASE": ["phosphatase"],
    "RECEPTOR_RTK": ["rtk"],
    "CYTOKINE_R": ["cytokine receptor"],
    "ION_VG": ["voltage-gated"],
    "ION_LG": ["ligand-gated"],
    "ION_TRP": ["trp"],
    "TRANSPORTER_ABC": ["abc transporter"],
    "TRANSPORTER_SLC": ["slc", "solute carrier"],
    "TF_ZF": ["zinc-finger", "zinc finger"],
    "TF_HOMEBOX": ["homeobox", "hox"],
    "TF_BHLH": ["bhlh"],
    "TF_BZIP": ["bzip"],
}

CLASS_SHORT = {
    "Transcription factor": "TF",
    "Receptor: GPCR": "GPCR",
    "Ion channel": "IonChannel",
    "Transporter": "Transporter",
    "Receptor: Nuclear": "NuclearR",
    "Enzyme": "Enzyme",
    "Receptor": "Receptor",
    "Other/Unknown": "Other",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _go_to_rule(go_id: str) -> str:
    """Convert a GO ID to a rule identifier component."""
    return "GO" + go_id.split(":", 1)[1].lstrip("0")


def _has_hint(patterns: Iterable[str], texts: Iterable[str]) -> bool:
    """Check if any of the patterns appear in any of the texts."""
    for text in texts:
        for pat in patterns:
            if pat in text:
                return True
    return False


def _first_hint(patterns: Iterable[str], texts: Iterable[str]) -> Optional[str]:
    """Return the first pattern found in any of the texts."""
    for text in texts:
        for pat in patterns:
            if pat in text:
                return pat
    return None


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def extract_signals(entry: Dict[str, Any]) -> SignalDict:
    """Extract relevant classification signals from a UniProt entry.

    Parameters
    ----------
    entry:
        UniProt JSON dictionary.

    Returns
    -------
    dict
        Dictionary mapping signal names to sets of string tokens. The keys are:
        ``go_ids``, ``go_terms``, ``ec_numbers``, ``keywords``, ``cross_refs``,
        ``xref_ids`` (``DB:ID``), ``features`` and ``texts``.
    """

    signals: SignalDict = {
        "go_ids": set(),
        "go_terms": set(),
        "ec_numbers": set(),
        "keywords": set(),
        "cross_refs": set(),
        "xref_ids": set(),
        "features": set(),
        "texts": set(),
    }

    # GO annotations ----------------------------------------------------
    for xref in entry.get("uniProtKBCrossReferences", []) or []:
        db = xref.get("database")
        if db:
            dbu = db.upper()
            signals["cross_refs"].add(dbu)
            xid = xref.get("id")
            if xid:
                signals["xref_ids"].add(f"{dbu}:{xid.upper()}")
        if db != "GO":
            continue
        go_id = xref.get("id")
        if not go_id:
            continue
        term = None
        for prop in xref.get("properties", []) or []:
            if prop.get("key") in {"GoTerm", "term"}:
                term = prop.get("value")
                break
        if isinstance(term, str) and term.startswith("F:"):
            signals["go_ids"].add(go_id)
            signals["go_terms"].add(term[2:].lower())

    # EC numbers --------------------------------------------------------
    desc = entry.get("proteinDescription", {}) or {}
    names = []
    if isinstance(desc, dict):
        names.append(desc.get("recommendedName"))
        names.extend(desc.get("alternativeNames", []) or [])
    for name in names:
        if not isinstance(name, dict):
            continue
        for ec in name.get("ecNumbers", []) or []:
            v = ec.get("value")
            if isinstance(v, str):
                signals["ec_numbers"].add(v)

    # Keywords ----------------------------------------------------------
    for kw in entry.get("keywords", []) or []:
        name = kw.get("name")
        if isinstance(name, str):
            signals["keywords"].add(name.lower())

    # Features ----------------------------------------------------------
    for feat in entry.get("features", []) or []:
        ftype = feat.get("type")
        if isinstance(ftype, str):
            signals["features"].add(ftype.upper())

    # Functional comments -----------------------------------------------
    for com in entry.get("comments", []) or []:
        if com.get("commentType") != "FUNCTION":
            continue
        for txt in com.get("texts", []) or []:
            v = txt.get("value")
            if isinstance(v, str):
                signals["texts"].add(v.lower())

    return signals


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_L1(signals: SignalDict) -> ClassificationResult:
    go = signals["go_ids"]
    texts = signals["keywords"] | signals["texts"] | signals["go_terms"]
    xrefs = signals["cross_refs"]
    evidence: List[str]

    # Transcription factor ------------------------------------------------
    if GO_TF in go or _has_hint(HINTS["TF"], texts):
        evidence = []
        if GO_TF in go:
            evidence.append(GO_TF)
            rule_id = f"L1.{CLASS_SHORT['Transcription factor']}.{_go_to_rule(GO_TF)}"
            conf = "high"
        else:
            hint = _first_hint(HINTS["TF"], texts) or "hint"
            evidence.append(f"TXT:{hint}")
            rule_id = f"L1.{CLASS_SHORT['Transcription factor']}.HINT"
            conf = "medium"
        return ClassificationResult("Transcription factor", rule_id, evidence, conf)

    # Receptor: GPCR ----------------------------------------------------
    if GO_GPCR in go or _has_hint(HINTS["GPCR"], texts):
        evidence = []
        if GO_GPCR in go:
            evidence.append(GO_GPCR)
        else:
            hint = _first_hint(HINTS["GPCR"], texts) or "hint"
            evidence.append(f"TXT:{hint}")
        rule_id = (
            f"L1.{CLASS_SHORT['Receptor: GPCR']}.{_go_to_rule(GO_GPCR)}"
            if GO_GPCR in go
            else f"L1.{CLASS_SHORT['Receptor: GPCR']}.HINT"
        )
        conf = "high" if GO_GPCR in go else "medium"
        return ClassificationResult("Receptor: GPCR", rule_id, evidence, conf)

    # Ion channel --------------------------------------------------------
    if GO_ION_CHANNEL in go or _has_hint(HINTS["ION_CHANNEL"], texts):
        evidence = []
        if GO_ION_CHANNEL in go:
            evidence.append(GO_ION_CHANNEL)
        else:
            hint = _first_hint(HINTS["ION_CHANNEL"], texts) or "hint"
            evidence.append(f"TXT:{hint}")
        rule_id = (
            f"L1.{CLASS_SHORT['Ion channel']}.{_go_to_rule(GO_ION_CHANNEL)}"
            if GO_ION_CHANNEL in go
            else f"L1.{CLASS_SHORT['Ion channel']}.HINT"
        )
        conf = "high" if GO_ION_CHANNEL in go else "medium"
        return ClassificationResult("Ion channel", rule_id, evidence, conf)

    # Transporter --------------------------------------------------------
    if (
        GO_TRANSPORTER in go
        or "TCDB" in xrefs
        or _has_hint(HINTS["TRANSPORTER"], texts)
    ):
        evidence = []
        if GO_TRANSPORTER in go:
            evidence.append(GO_TRANSPORTER)
            rule_id = f"L1.{CLASS_SHORT['Transporter']}.{_go_to_rule(GO_TRANSPORTER)}"
            conf = "high"
        elif "TCDB" in xrefs:
            evidence.append("XREF:TCDB")
            rule_id = f"L1.{CLASS_SHORT['Transporter']}.TCDB"
            conf = "high"
        else:
            hint = _first_hint(HINTS["TRANSPORTER"], texts) or "hint"
            evidence.append(f"TXT:{hint}")
            rule_id = f"L1.{CLASS_SHORT['Transporter']}.HINT"
            conf = "medium"
        return ClassificationResult("Transporter", rule_id, evidence, conf)

    # Receptor: Nuclear --------------------------------------------------
    if GO_NUCLEAR_R in go or _has_hint(HINTS["NUCLEAR_R"], texts):
        evidence = []
        if GO_NUCLEAR_R in go:
            evidence.append(GO_NUCLEAR_R)
            rule_id = (
                f"L1.{CLASS_SHORT['Receptor: Nuclear']}.{_go_to_rule(GO_NUCLEAR_R)}"
            )
            conf = "high"
        else:
            hint = _first_hint(HINTS["NUCLEAR_R"], texts) or "hint"
            evidence.append(f"TXT:{hint}")
            rule_id = f"L1.{CLASS_SHORT['Receptor: Nuclear']}.HINT"
            conf = "medium"
        return ClassificationResult("Receptor: Nuclear", rule_id, evidence, conf)

    # Enzyme -------------------------------------------------------------
    if (
        signals["ec_numbers"]
        or GO_CATALYTIC in go
        or _has_hint(HINTS["KINASE"], texts)
        or _has_hint(HINTS["PROTEASE"], texts)
        or "MEROPS" in xrefs
    ):
        evidence = []
        if signals["ec_numbers"]:
            evidence.extend(f"EC:{e}" for e in sorted(signals["ec_numbers"]))
            rule_id = f"L1.{CLASS_SHORT['Enzyme']}.EC"
            conf = "high"
        elif "MEROPS" in xrefs:
            evidence.append("XREF:MEROPS")
            rule_id = f"L1.{CLASS_SHORT['Enzyme']}.MEROPS"
            conf = "high"
        elif GO_CATALYTIC in go:
            evidence.append(GO_CATALYTIC)
            rule_id = f"L1.{CLASS_SHORT['Enzyme']}.{_go_to_rule(GO_CATALYTIC)}"
            conf = "medium"
        else:
            hint = (
                _first_hint(HINTS["KINASE"], texts)
                or _first_hint(HINTS["PROTEASE"], texts)
                or "hint"
            )
            evidence.append(f"TXT:{hint}")
            rule_id = f"L1.{CLASS_SHORT['Enzyme']}.HINT"
            conf = "medium"
        return ClassificationResult("Enzyme", rule_id, evidence, conf)

    # Receptor (generic) -------------------------------------------------
    if any("receptor" in t for t in texts):
        rule_id = f"L1.{CLASS_SHORT['Receptor']}.HINT"
        evidence = ["TXT:receptor"]
        return ClassificationResult("Receptor", rule_id, evidence, "medium")

    # Other/Unknown ------------------------------------------------------
    evidence = []
    if "TRANSMEM" in signals["features"] or "TRANSMEMBRANE" in signals["features"]:
        evidence.append("FEAT:TRANSMEM")
        conf = "low"
    else:
        conf = "low"
    rule_id = f"L1.{CLASS_SHORT['Other/Unknown']}.NA"
    return ClassificationResult("Other/Unknown", rule_id, evidence, conf)


def classify_L1(signals: SignalDict) -> str:
    """Classifies the protein into a coarse L1 class.

    Args:
        signals: A dictionary of signals extracted from the UniProt entry.

    Returns:
        The L1 classification label.
    """
    return _classify_L1(signals).label


# ---------------------------------------------------------------------------


def _classify_L2(signals: SignalDict, l1: str) -> ClassificationResult:
    go = signals["go_ids"]
    texts = signals["keywords"] | signals["texts"] | signals["go_terms"]
    xrefs = signals["cross_refs"]

    if l1 == "Enzyme":
        if (
            GO_KINASE in go
            or _has_hint(HINTS["KINASE"], texts)
            or "PFAM:PF00069" in signals["xref_ids"]
        ):
            evidence = []
            if GO_KINASE in go:
                evidence.append(GO_KINASE)
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Kinase.{_go_to_rule(GO_KINASE)}"
                conf = "high"
            elif "PFAM:PF00069" in signals["xref_ids"]:
                evidence.append("XREF:PFAM:PF00069")
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Kinase.PF00069"
                conf = "high"
            else:
                hint = _first_hint(HINTS["KINASE"], texts) or "hint"
                evidence.append(f"TXT:{hint}")
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Kinase.HINT"
                conf = "medium"
            return ClassificationResult("Kinase", rule_id, evidence, conf)

        if (
            GO_PROTEASE in go
            or "MEROPS" in xrefs
            or _has_hint(HINTS["PROTEASE"], texts)
        ):
            evidence = []
            if GO_PROTEASE in go:
                evidence.append(GO_PROTEASE)
                rule_id = (
                    f"L2.{CLASS_SHORT['Enzyme']}.Protease.{_go_to_rule(GO_PROTEASE)}"
                )
                conf = "high"
            elif "MEROPS" in xrefs:
                evidence.append("XREF:MEROPS")
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Protease.MEROPS"
                conf = "high"
            else:
                hint = _first_hint(HINTS["PROTEASE"], texts) or "hint"
                evidence.append(f"TXT:{hint}")
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Protease.HINT"
                conf = "medium"
            return ClassificationResult("Protease", rule_id, evidence, conf)

        if any(g in go for g in GO_PHOSPHATASE_IDS) or _has_hint(
            HINTS["PHOSPHATASE"], texts
        ):
            evidence = []
            g = next((g for g in GO_PHOSPHATASE_IDS if g in go), None)
            if g:
                evidence.append(g)
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Phosphatase.{_go_to_rule(g)}"
                conf = "high"
            else:
                hint = _first_hint(HINTS["PHOSPHATASE"], texts) or "hint"
                evidence.append(f"TXT:{hint}")
                rule_id = f"L2.{CLASS_SHORT['Enzyme']}.Phosphatase.HINT"
                conf = "medium"
            return ClassificationResult("Phosphatase", rule_id, evidence, conf)

        return ClassificationResult(
            "Enzyme: Other", f"L2.{CLASS_SHORT['Enzyme']}.Other", [], "medium"
        )

    if l1 == "Receptor":
        if GO_KINASE in go and any("receptor" in t for t in texts):
            evidence = [GO_KINASE]
            rule_id = f"L2.{CLASS_SHORT['Receptor']}.RTK.{_go_to_rule(GO_KINASE)}"
            return ClassificationResult("RTK", rule_id, evidence, "high")
        if _has_hint(HINTS["CYTOKINE_R"], texts):
            hint = _first_hint(HINTS["CYTOKINE_R"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Receptor']}.Cytokine.HINT"
            return ClassificationResult(
                "Cytokine receptor", rule_id, evidence, "medium"
            )
        return ClassificationResult(
            "Receptor: Other", f"L2.{CLASS_SHORT['Receptor']}.Other", [], "medium"
        )

    if l1 == "Ion channel":
        if _has_hint(HINTS["ION_VG"], texts):
            hint = _first_hint(HINTS["ION_VG"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Ion channel']}.VoltageGated.HINT"
            return ClassificationResult("Voltage-gated", rule_id, evidence, "high")
        if _has_hint(HINTS["ION_LG"], texts):
            hint = _first_hint(HINTS["ION_LG"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Ion channel']}.LigandGated.HINT"
            return ClassificationResult("Ligand-gated", rule_id, evidence, "high")
        if _has_hint(HINTS["ION_TRP"], texts):
            hint = _first_hint(HINTS["ION_TRP"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Ion channel']}.TRP.HINT"
            return ClassificationResult("TRP channel", rule_id, evidence, "high")
        return ClassificationResult(
            "Ion channel: Other", f"L2.{CLASS_SHORT['Ion channel']}.Other", [], "medium"
        )

    if l1 == "Transporter":
        if _has_hint(HINTS["TRANSPORTER_ABC"], texts) or any(
            xid.startswith("TCDB:3.A.1") for xid in signals["xref_ids"]
        ):
            evidence = []
            if any(xid.startswith("TCDB:3.A.1") for xid in signals["xref_ids"]):
                evidence.append("XREF:TCDB")
            else:
                hint = _first_hint(HINTS["TRANSPORTER_ABC"], texts) or "hint"
                evidence.append(f"TXT:{hint}")
            rule_id = f"L2.{CLASS_SHORT['Transporter']}.ABC"
            return ClassificationResult("ABC", rule_id, evidence, "high")
        if _has_hint(HINTS["TRANSPORTER_SLC"], texts):
            hint = _first_hint(HINTS["TRANSPORTER_SLC"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Transporter']}.SLC"
            return ClassificationResult("SLC", rule_id, evidence, "high")
        return ClassificationResult(
            "Transporter: Other", f"L2.{CLASS_SHORT['Transporter']}.Other", [], "medium"
        )

    if l1 == "Transcription factor":
        if _has_hint(HINTS["TF_ZF"], texts):
            hint = _first_hint(HINTS["TF_ZF"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Transcription factor']}.ZincFinger"
            return ClassificationResult("Zinc finger", rule_id, evidence, "high")
        if _has_hint(HINTS["TF_HOMEBOX"], texts):
            hint = _first_hint(HINTS["TF_HOMEBOX"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Transcription factor']}.Homeobox"
            return ClassificationResult("Homeobox", rule_id, evidence, "high")
        if _has_hint(HINTS["TF_BHLH"], texts):
            hint = _first_hint(HINTS["TF_BHLH"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Transcription factor']}.bHLH"
            return ClassificationResult("bHLH", rule_id, evidence, "high")
        if _has_hint(HINTS["TF_BZIP"], texts):
            hint = _first_hint(HINTS["TF_BZIP"], texts) or "hint"
            evidence = [f"TXT:{hint}"]
            rule_id = f"L2.{CLASS_SHORT['Transcription factor']}.bZIP"
            return ClassificationResult("bZIP", rule_id, evidence, "high")
        return ClassificationResult(
            "TF: Other", f"L2.{CLASS_SHORT['Transcription factor']}.Other", [], "medium"
        )

    if l1 == "Receptor: Nuclear":
        for token in texts:
            m = re.search(r"(NR\d+[A-Z])", token)
            if m:
                fam = m.group(1).upper()
                return ClassificationResult(
                    fam,
                    f"L2.{CLASS_SHORT['Receptor: Nuclear']}.{fam}",
                    [f"TXT:{fam}"],
                    "high",
                )
            for fam in ["PPAR", "RXR"]:
                if fam.lower() in token:
                    return ClassificationResult(
                        fam,
                        f"L2.{CLASS_SHORT['Receptor: Nuclear']}.{fam}",
                        [f"TXT:{fam}"],
                        "high",
                    )
        return ClassificationResult(
            "Nuclear receptor: Other",
            f"L2.{CLASS_SHORT['Receptor: Nuclear']}.Other",
            [],
            "medium",
        )

    return ClassificationResult("NA", "", [], "")


def classify_L2(signals: SignalDict, l1: str) -> str:
    """Classifies the protein into an L2 class, given the L1 class.

    Args:
        signals: A dictionary of signals extracted from the UniProt entry.
        l1: The L1 classification label.

    Returns:
        The L2 classification label.
    """
    return _classify_L2(signals, l1).label


# ---------------------------------------------------------------------------


def _classify_L3(signals: SignalDict, l1: str, l2: str) -> ClassificationResult:
    texts = signals["keywords"] | signals["texts"] | signals["go_terms"]
    if l1 == "Enzyme" and l2 == "Kinase":
        for fam in ["RTK", "CAMK", "AGC", "CMGC"]:
            if fam.lower() in " ".join(texts):
                return ClassificationResult(
                    fam, f"L3.Enzyme.Kinase.{fam}", [f"TXT:{fam}"], "high"
                )
        return ClassificationResult("NA", "", [], "")
    if l1 == "Enzyme" and l2 == "Protease":
        merops_ids = [
            x.split(":", 1)[1] for x in signals["xref_ids"] if x.startswith("MEROPS:")
        ]
        if merops_ids:
            fam = merops_ids[0][:3]
            return ClassificationResult(
                fam,
                f"L3.Enzyme.Protease.{fam}",
                [f"XREF:MEROPS:{merops_ids[0]}"],
                "high",
            )
        return ClassificationResult("NA", "", [], "")
    if l1 == "Transcription factor" and l2 == "Zinc finger":
        if any("krab" in t for t in texts):
            return ClassificationResult(
                "KRAB", "L3.TF.ZincFinger.KRAB", ["TXT:krab"], "high"
            )
        return ClassificationResult("NA", "", [], "")
    if l1 == "Transporter" and l2 == "ABC":
        for token in texts:
            m = re.search(r"(ABC[ A-Z]?)([A-Z])", token)
            if m:
                fam = m.group(1).replace(" ", "")
                return ClassificationResult(
                    fam, f"L3.Transporter.ABC.{fam}", [f"TXT:{fam}"], "high"
                )
        return ClassificationResult("NA", "", [], "")
    if l1 == "Transporter" and l2 == "SLC":
        for token in texts:
            m = re.search(r"(SLC\d+[A-Z]?)", token)
            if m:
                fam = m.group(1).upper()
                return ClassificationResult(
                    fam, f"L3.Transporter.SLC.{fam}", [f"TXT:{fam}"], "high"
                )
        return ClassificationResult("NA", "", [], "medium")
    if l1 == "Ion channel" and l2 == "TRP channel":
        for fam in ["TRPV", "TRPA", "TRPM", "TRPC"]:
            if fam.lower() in " ".join(texts):
                return ClassificationResult(
                    fam, f"L3.IonChannel.TRP.{fam}", [f"TXT:{fam}"], "high"
                )
        return ClassificationResult("NA", "", [], "medium")
    return ClassificationResult("NA", "", [], "")


def classify_L3(signals: SignalDict, l1: str, l2: str) -> str:
    """Classifies the protein into an L3 class, given the L1 and L2 classes.

    Args:
        signals: A dictionary of signals extracted from the UniProt entry.
        l1: The L1 classification label.
        l2: The L2 classification label.

    Returns:
        The L3 classification label.
    """
    return _classify_L3(signals, l1, l2).label


# ---------------------------------------------------------------------------


def classify_protein(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Classifies a UniProt entry into hierarchical protein classes.

    Args:
        entry: A UniProt JSON dictionary.

    Returns:
        A dictionary with the following keys: `protein_class_L1`,
        `protein_class_L2`, `protein_class_L3`, `rule_id`, `evidence`,
        and `confidence`.
    """
    signals = extract_signals(entry)
    l1_res = _classify_L1(signals)
    l2_res = _classify_L2(signals, l1_res.label)
    l3_res = _classify_L3(signals, l1_res.label, l2_res.label)
    rule = l3_res.rule_id or l2_res.rule_id or l1_res.rule_id
    evidence = l3_res.evidence or l2_res.evidence or l1_res.evidence
    confidence = l3_res.confidence or l2_res.confidence or l1_res.confidence

    return {
        "protein_class_L1": l1_res.label,
        "protein_class_L2": l2_res.label if l1_res.label != "Other/Unknown" else "NA",
        "protein_class_L3": (
            l3_res.label
            if l2_res.label
            not in {
                "NA",
                "Enzyme: Other",
                "Transporter: Other",
                "Ion channel: Other",
                "TF: Other",
                "Receptor: Other",
                "Nuclear receptor: Other",
            }
            else "NA"
        ),
        "rule_id": rule,
        "evidence": evidence,
        "confidence": confidence,
    }
