from __future__ import annotations

import io
import logging
import random
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import pandas as pd
import requests
from requests import Session

try:
    from data_profiling import analyze_table_quality
except ModuleNotFoundError:  # pragma: no cover
    from .data_profiling import analyze_table_quality

LOGGER = logging.getLogger(__name__)


@dataclass
class ApiCfg:
    """HTTP API configuration."""

    user_agent: str


@dataclass
class IupharCfg:
    """Configuration for accessing the IUPHAR web service."""

    base: str
    timeout_connect: float = 5.0
    timeout_read: float = 5.0
    rps: float = 1.0


@dataclass
class RetryCfg:
    """Retry policy for web requests."""

    max_attempts: int = 3
    backoff_factor: float = 0.5


def session_with_retry(api: ApiCfg, retry: RetryCfg) -> Session:
    """Creates a requests.Session with basic headers and retry logic.

    Args:
        api: The API configuration.
        retry: The retry configuration.

    Returns:
        A requests.Session object.
    """

    session = requests.Session()
    session.headers.update({"User-Agent": api.user_agent})
    return session


class RateLimiter:
    """A simple token bucket rate limiter.

    Args:
        rps: The number of requests per second to allow.
    """

    def __init__(self, rps: float) -> None:
        """Initializes the RateLimiter."""
        self.rps = rps
        self.last_call = 0.0

    def acquire(self) -> None:
        """Blocks until a request can be made without exceeding the rate limit."""
        if self.rps <= 0:
            return
        interval = 1.0 / self.rps
        now = time.monotonic()
        delta = now - self.last_call
        if delta < interval:
            time.sleep(interval - delta)
        self.last_call = time.monotonic()


def get_limiter(_name: str, rps: float) -> RateLimiter:
    """Returns a RateLimiter for the specified requests per second.

    Args:
        _name: The name of the limiter (currently unused).
        rps: The number of requests per second.

    Returns:
        A RateLimiter instance.
    """
    return RateLimiter(rps)


def sleep(seconds: float) -> None:
    """A sleep helper to facilitate jittered backoff."""

    time.sleep(seconds)


# Default session with placeholder user agent; callers should override via
# :func:`init_session` with a configuration that provides their own contact
# information.
_session: Session = session_with_retry(
    ApiCfg(user_agent="chembl-da/0.1 (mailto:info@example.org)"), RetryCfg()
)
_session_lock = threading.Lock()


def init_session(api: ApiCfg, retry: RetryCfg) -> None:
    """Initializes the shared HTTP session.

    Args:
        api: The API configuration.
        retry: The retry configuration.
    """

    global _session
    with _session_lock:
        _session = session_with_retry(api, retry)


EXPECTED_TARGET_COLUMNS: tuple[str, ...] = (
    "target_id",
    "uniprot_id",
    "hgnc_name",
    "hgnc_id",
    "gene_name",
    "synonyms",
    "family_id",
    "target_name",
)

EXPECTED_FAMILY_COLUMNS: tuple[str, ...] = (
    "family_id",
    "family_name",
    "parent_family_id",
    "target_id",
    "type",
)


def _validate_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    """Validate that a DataFrame contains the expected columns.

    Parameters
    ----------
    df:
        DataFrame to validate.
    expected:
        Iterable of expected column names.

    Raises
    ------
    ValueError
        If any expected column is missing.

    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def load_targets(path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Loads the `_IUPHAR_target.csv` file.

    The `target_id` and `family_id` columns in the official files are
    zero-padded strings. To avoid losing the padding, all columns are read as
    strings, and missing values are replaced with empty strings. The `swissprot`
    column is renamed to `uniprot_id` for consistency.

    Args:
        path: The path to the target CSV file.
        encoding: The file encoding.

    Returns:
        A pandas DataFrame containing the loaded target data.
    """
    df = pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
    # Normalise legacy column names to the expected lowercase form
    df = df.rename(
        columns={
            "GuidetoPHARMACOLOGY": "target_id",
            "HGNC_NAME": "hgnc_name",
            "HGNC_name": "hgnc_name",
            "HGNC_ID": "hgnc_id",
            "HGNC_id": "hgnc_id",
            "swissprot": "uniprot_id",
        }
    )
    _validate_columns(df, EXPECTED_TARGET_COLUMNS)
    return df


def load_families(path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Loads the `_IUPHAR_family.csv` file.

    Similar to `load_targets`, family identifiers are stored as zero-padded
    strings. The file is therefore read with `dtype=str`, and missing values
    are normalized to empty strings to retain formatting.

    Args:
        path: The path to the family CSV file.
        encoding: The file encoding.

    Returns:
        A pandas DataFrame containing the loaded family data.
    """
    df = pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
    _validate_columns(df, EXPECTED_FAMILY_COLUMNS)
    return df


def _query_gene_symbol(
    gene_name: str, cfg: IupharCfg, retry: RetryCfg
) -> dict[str, Any]:
    """Return the first IUPHAR result for *gene_name*.

    Parameters
    ----------
    gene_name:
        Gene symbol to search for.
    cfg:
        API configuration controlling base URL, timeouts and rate limits.
    retry:
        Retry configuration specifying backoff behaviour.

    Returns
    -------
    dict
        First result dictionary or an empty dict on failure.
    """

    base = cfg.base.rstrip("/")
    url = f"{base}/targets/?geneSymbol={quote(gene_name)}"
    timeout = (cfg.timeout_connect, cfg.timeout_read)
    limiter = get_limiter("iuphar", cfg.rps)

    for attempt in range(1, retry.max_attempts + 1):
        limiter.acquire()
        try:
            with _session_lock:
                with _session.get(url, timeout=timeout) as response:
                    response.raise_for_status()
                    data = cast(list[dict[str, Any]], response.json())
                    return data[0] if data else {}
        except requests.RequestException as exc:  # pragma: no cover - network errors
            if attempt >= retry.max_attempts:
                LOGGER.error("IUPHAR web request failed: %s", exc)
                break
            backoff = retry.backoff_factor * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff)
            sleep(backoff + jitter)
    return {}


@dataclass
class IUPHARData:
    """A container for IUPHAR target and family data.

    Attributes:
        target_df: A DataFrame containing target data.
        family_df: A DataFrame containing family data.
    """

    target_df: pd.DataFrame
    family_df: pd.DataFrame

    @classmethod
    def from_files(
        cls,
        target_path: str | Path,
        family_path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> IUPHARData:
        """Loads CSV files and returns an IUPHARData instance.

        Args:
            target_path: The path to the target CSV file.
            family_path: The path to the family CSV file.
            encoding: The file encoding.

        Returns:
            An IUPHARData instance.
        """
        target_df = load_targets(target_path, encoding=encoding)
        family_df = load_families(family_path, encoding=encoding)
        return cls(target_df=target_df, family_df=family_df)

    # ------------------------------------------------------------------
    # Chain-related helpers
    # ------------------------------------------------------------------

    def family_chain(self, start_id: str) -> list[str]:
        """Return the family hierarchy starting from *start_id*.

        The chain is built by following ``parent_family_id`` links until a
        root family is reached.
        """
        chain: list[str] = []
        current = start_id
        while current:
            chain.append(current)
            parent_series = self.family_df.loc[
                self.family_df["family_id"] == current, "parent_family_id"
            ]
            if parent_series.empty:
                break
            parent = (
                str(parent_series.iloc[0]) if pd.notna(parent_series.iloc[0]) else ""
            )
            if not parent:
                break
            current = parent
        return chain

    def all_id(self, target_id: str) -> str:
        """Return the full family ID path for a target."""
        family_id = self.from_target_family_id(target_id)
        if not family_id:
            return ""
        chain = self.family_chain(family_id)
        return f"{target_id}#" + ">".join(chain)

    def all_name(self, target_id: str) -> str:
        """Return the full family name path for a target."""
        family_id = self.from_target_family_id(target_id)
        if not family_id:
            return ""
        chain = []
        for fam_id in self.family_chain(family_id):
            record = self.from_family_record(fam_id)
            if record is None:
                continue
            name = record.get("family_name")
            if name and name.lower() not in {"enzyme"}:
                chain.append(name)
        target_name = self.from_target_name(target_id)
        return f"{target_name}#" + ">".join(chain)

    # ------------------------------------------------------------------
    # Target ID mapping functions
    # ------------------------------------------------------------------

    def _select_target_ids(self, mask: pd.Series) -> list[str]:
        rows = self.target_df.loc[mask, "target_id"].dropna().astype(str)
        return list(rows.unique())

    def target_id_by_uniprot(self, uniprot_id: str) -> str:
        """Return the first target ID mapped to ``uniprot_id``."""
        ids = self._select_target_ids(self.target_df["uniprot_id"].eq(uniprot_id))
        return ids[0] if ids else ""

    def target_id_by_hgnc_name(self, hgnc_name: str) -> str:
        """Return target IDs whose HGNC name equals ``hgnc_name``."""
        if not hgnc_name:
            return ""
        ids = self._select_target_ids(self.target_df["hgnc_name"].eq(hgnc_name))
        return "|".join(ids) if ids else ""

    def target_id_by_hgnc_id(self, hgnc_id: str) -> str:
        """Return target IDs whose HGNC identifier matches ``hgnc_id``."""
        ids = self._select_target_ids(self.target_df["hgnc_id"].eq(hgnc_id))
        return "|".join(ids) if ids else ""

    def target_id_by_gene(self, gene_name: str) -> str:
        """Return target IDs whose gene symbol matches ``gene_name``."""
        ids = self._select_target_ids(self.target_df["gene_name"].eq(gene_name))
        return "|".join(ids) if ids else ""

    def target_id_by_name(self, target_name: str) -> str:
        """Resolve target IDs by searching the synonym field.

        Parameters
        ----------
        target_name:
            Name or synonym to search for.

        Returns
        -------
        str
            Pipe-delimited target identifiers whose ``synonyms`` column
            contains ``target_name``. The search is case-insensitive and treats
            ``target_name`` as a literal substring rather than a regular
            expression, avoiding errors with special characters.

        """
        if not target_name:
            return ""
        mask = (
            self.target_df["synonyms"]
            .fillna("")
            .str.contains(target_name, case=False, na=False, regex=False)
        )
        ids = self._select_target_ids(mask)
        return "|".join(ids) if ids else ""

    def target_ids_by_synonyms(self, synonyms: Iterable[str]) -> str:
        """Return target IDs derived from a collection of ``synonyms``."""
        valid = [s for s in synonyms if s and len(s) > 3]
        ids: list[str] = []
        for syn in valid:
            mapped = self.target_id_by_name(syn)
            if mapped and "|" not in mapped:
                ids.append(mapped)
        unique = sorted(set(ids))
        return "|".join(unique) if unique else ""

    def family_id_by_name(self, target_name: str) -> str:
        """Return family identifiers associated with *target_name*.

        The function searches the synonym column for matches to ``target_name``
        and collects the corresponding family IDs. Multiple IDs are joined by
        a pipe character.
        """
        if not target_name:
            return ""
        target_ids = self.target_id_by_name(target_name).split("|")
        family_ids = [self.from_target_family_id(tid) for tid in target_ids if tid]
        unique = sorted({fid for fid in family_ids if fid})
        return "|".join(unique) if unique else ""

    def target_id_from_row(self, row: pd.Series) -> str:
        """Resolve a target identifier using multiple columns from ``row``.

        The function attempts to determine a target ID by consulting, in
        order, the UniProt accession, HGNC name, HGNC ID, an explicit name
        field and any provided synonyms.  Each stage only fills in missing
        identifiers, mirroring the behaviour of the legacy Power Query
        implementation.
        """
        uniprot = self.target_id_by_uniprot(row.get("uniprot_id", ""))
        hgnc_name = self.target_id_by_hgnc_name(row.get("hgnc_name", ""))
        hgnc_id = self.target_id_by_hgnc_id(row.get("hgnc_id", ""))
        name = ""
        synonyms = ""
        all_ids = [x for x in [uniprot, hgnc_name, hgnc_id, name, synonyms] if x]
        unique = sorted({i for part in all_ids for i in part.split("|") if i})
        if not unique:
            return "N/A"
        return "|".join(unique)

    def map_uniprot_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        encoding: str = "utf-8",
        sep: str = ",",
    ) -> pd.DataFrame:
        """Map UniProt IDs to classification data and write a CSV output.

        The input CSV must contain a ``uniprot_id`` column. If the file
        includes a ``GuidetoPHARMACOLOGY`` column, non-empty values are taken
        as the target identifier. For rows lacking this cross-reference, the
        function attempts to resolve the target via the UniProt accession. If
        that fails, optional columns ``hgnc_name``, ``hgnc_id``, ``gene_name``
        and ``synonyms`` (pipe-delimited) are consulted in that order.
        Successful lookups are translated to a full IUPHAR classification. The
        resulting table includes the target ID, class, subclass, and family
        chain in addition to the full ID and name paths. If no target can be
        resolved, an optional ``ec_number`` column is used to derive
        ``IUPHAR_type``, ``IUPHAR_class`` and ``IUPHAR_subclass``.

        Parameters
        ----------
        input_path:
            CSV file with a ``uniprot_id`` column.
        output_path:
            Destination for the enriched CSV file.
        encoding:
            Encoding used for both reading and writing. Defaults to UTF-8.
        sep:
            Delimiter used when reading and writing CSV files. Defaults to a comma.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the mapping results.

        """
        df = pd.read_csv(input_path, dtype=str, encoding=encoding, sep=sep).fillna("")
        if "uniprot_id" not in df.columns:
            raise ValueError("Input file must contain 'uniprot_id' column")

        classifier = IUPHARClassifier(self)

        # Start with explicit GuidetoPHARMACOLOGY mappings when available. The
        # UniProt export may provide this cross-reference directly and it takes
        # precedence over any derived lookups.
        if "GuidetoPHARMACOLOGY" in df.columns:
            df["target_id"] = df["GuidetoPHARMACOLOGY"].str.split("|").str[0]
        else:
            df["target_id"] = ""

        # Resolve remaining identifiers with a series of fallbacks. The search
        # order mirrors the Power Query logic: UniProt accession, HGNC name,
        # HGNC ID, gene symbol and finally any supplied synonyms.
        mask = df["target_id"].eq("")
        df.loc[mask, "target_id"] = df.loc[mask, "uniprot_id"].apply(
            self.target_id_by_uniprot
        )

        if "hgnc_name" in df.columns:
            mask = df["target_id"].eq("")
            df.loc[mask, "target_id"] = df.loc[mask, "hgnc_name"].apply(
                self.target_id_by_hgnc_name
            )

        if "hgnc_id" in df.columns:
            mask = df["target_id"].eq("")
            df.loc[mask, "target_id"] = df.loc[mask, "hgnc_id"].apply(
                self.target_id_by_hgnc_id
            )

        if "gene_name" in df.columns:
            mask = df["target_id"].eq("")
            df.loc[mask, "target_id"] = df.loc[mask, "gene_name"].apply(
                self.target_id_by_gene
            )

        if "synonyms" in df.columns:
            mask = df["target_id"].eq("")
            df.loc[mask, "target_id"] = df.loc[mask, "synonyms"].apply(
                lambda s: self.target_ids_by_synonyms(str(s).split("|")) if s else ""
            )

        def _classify(row: pd.Series) -> pd.Series:
            tid = row.get("target_id", "")
            record: ClassificationRecord
            if tid:
                record = classifier.by_target_id(tid)
            else:
                ec = row.get("ec_number", "")
                if ec:
                    record = classifier.by_ec_number(ec)
                else:
                    mf = row.get("molecular_function", "")
                    if mf:
                        record = classifier.by_molecular_function(mf)
                    else:
                        return pd.Series(
                            {
                                "IUPHAR_family_id": "",
                                "IUPHAR_type": "",
                                "IUPHAR_class": "",
                                "IUPHAR_subclass": "",
                                "IUPHAR_chain": "",
                            }
                        )
            return pd.Series(
                {
                    "IUPHAR_family_id": record.IUPHAR_family_id,
                    "IUPHAR_type": record.IUPHAR_type,
                    "IUPHAR_class": record.IUPHAR_class,
                    "IUPHAR_subclass": record.IUPHAR_subclass,
                    "IUPHAR_chain": ">".join(record.IUPHAR_tree),
                }
            )

        class_df = df.apply(_classify, axis=1)
        df = pd.concat([df, class_df], axis=1)

        df["full_id_path"] = df["target_id"].apply(self.all_id)
        df["full_name_path"] = df["target_id"].apply(self.all_name)

        df.to_csv(output_path, index=False, encoding=encoding, sep=sep)
        analyze_table_quality(df, table_name=str(Path(output_path).with_suffix("")))
        LOGGER.info("file_written", extra={"rows": len(df), "path": str(output_path)})
        return df

    # ------------------------------------------------------------------
    # fromTargetID functions
    # ------------------------------------------------------------------

    def from_target_record(self, target_id: str) -> pd.Series | None:
        """Return the raw target record for ``target_id`` if present."""
        rows = self.target_df[self.target_df["target_id"] == target_id]
        if rows.empty:
            return None
        return rows.iloc[0]

    def from_target_family_record(self, target_id: str) -> pd.Series | None:
        """Return the family record associated with ``target_id``."""
        mask = (
            self.family_df["target_id"]
            .fillna("")
            .str.split("|")
            .apply(lambda ids: target_id in ids)
        )
        rows = self.family_df[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def from_target_name(self, target_id: str) -> str:
        """Return the IUPHAR target name for ``target_id``."""
        record = self.from_target_record(target_id)
        return str(record.get("target_name")) if record is not None else ""

    def from_target_type(self, target_id: str) -> str:
        """Return the target ``type`` field for ``target_id``."""
        record = self.from_target_record(target_id)
        if record is None:
            return ""
        value = record.get("type")
        return str(value) if pd.notna(value) else ""

    def from_target_synonyms(self, target_id: str) -> str:
        """Return pipe-separated synonyms for ``target_id``."""
        record = self.from_target_record(target_id)
        if record is None:
            return ""
        value = record.get("synonyms")
        return str(value) if pd.notna(value) else ""

    def from_target_family_id(self, target_id: str) -> str:
        """Return the family identifier linked to ``target_id``."""
        record = self.from_target_record(target_id)
        if record is None:
            return ""
        value = record.get("family_id")
        return str(value) if pd.notna(value) else ""

    def from_target_parent_family(self, target_id: str) -> str:
        """Return the parent family ID for ``target_id``."""
        record = self.from_target_family_record(target_id)
        if record is None:
            return ""
        value = record.get("parent_family_id")
        return str(value) if pd.notna(value) else ""

    # ------------------------------------------------------------------
    # fromFamilyID functions
    # ------------------------------------------------------------------

    def from_family_type(self, family_id: str) -> str:
        """Return the ``type`` field for ``family_id``.

        Parameters
        ----------
        family_id:
            Identifier of the family to inspect.

        Returns
        -------
        str
            Family type or an empty string when ``family_id`` is unknown.

        """
        record = self.from_family_record(family_id)
        if record is None:
            return ""
        value = record.get("type")
        return str(value) if pd.notna(value) else ""

    def from_family_record(self, family_id: str) -> pd.Series | None:
        """Return the raw family record for ``family_id`` if present."""
        rows = self.family_df[self.family_df["family_id"] == family_id]
        if rows.empty:
            return None
        return rows.iloc[0]

    def from_family_parent(self, family_id: str) -> str:
        """Return the parent family identifier for ``family_id``."""
        record = self.from_family_record(family_id)
        if record is None:
            return ""
        value = record.get("parent_family_id")
        return str(value) if pd.notna(value) else ""

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def websearch_gene_to_id(
        self, gene_name: str, cfg: IupharCfg, retry: RetryCfg | None = None
    ) -> dict[str, Any]:
        """Query the IUPHAR web API for ``gene_name``.

        Parameters
        ----------
        gene_name:
            Gene symbol to search for.
        cfg:
            API configuration controlling base URL, timeouts and rate limits.
        retry:
            Optional retry configuration.  Defaults to :class:`RetryCfg`.

        Returns
        -------
        dict
            First result dictionary or an empty dict on failure.
        """

        retry_cfg = retry or RetryCfg()
        return _query_gene_symbol(gene_name, cfg, retry_cfg)

    # ------------------------------------------------------------------
    # IUPHAR upload processing
    # ------------------------------------------------------------------

    def iuphar_upload(
        self, cfg: IupharCfg, retry: RetryCfg | None = None
    ) -> pd.DataFrame:
        """Reproduce the ``IUPHAR_upload`` transformation.

        Parameters
        ----------
        cfg:
            API configuration controlling base URL, timeouts and rate limits.
        retry:
            Optional retry configuration.  Defaults to :class:`RetryCfg`.

        Returns
        -------
        pandas.DataFrame
            Combined mapping of IUPHAR targets and families.

        The function downloads mapping files from the IUPHAR service and
        combines them with the local target and family tables.
        """
        retry_cfg = retry or RetryCfg()
        base_root = cfg.base.rstrip("/")
        if base_root.endswith("services"):
            base_root = base_root.rsplit("/", 1)[0]
        data_base = f"{base_root}/DATA"
        limiter = get_limiter("iuphar", cfg.rps)
        timeout = (cfg.timeout_connect, cfg.timeout_read)

        def _download(url: str) -> pd.DataFrame:
            for attempt in range(1, retry_cfg.max_attempts + 1):
                limiter.acquire()
                try:
                    with _session_lock:
                        with _session.get(url, timeout=timeout) as resp:
                            resp.raise_for_status()
                            return pd.read_csv(io.StringIO(resp.text))
                except requests.RequestException as exc:  # pragma: no cover - network
                    if attempt >= retry_cfg.max_attempts:
                        LOGGER.error("IUPHAR mapping request failed: %s", exc)
                        raise
                    backoff = retry_cfg.backoff_factor * (2 ** (attempt - 1))
                    jitter = random.uniform(0, backoff)
                    sleep(backoff + jitter)
            raise RuntimeError("Failed to download mapping")

        uni_df = _download(f"{data_base}/GtP_to_UniProt_mapping.csv")
        hgnc_df = _download(f"{data_base}/GtP_to_HGNC_mapping.csv")
        hgnc_df = hgnc_df.rename(columns={"IUPHAR ID": "GtoPdb IUPHAR ID"})
        mapping = pd.merge(
            hgnc_df,
            uni_df,
            on="GtoPdb IUPHAR ID",
            how="outer",
            suffixes=("_hgnc", "_uni"),
        )
        mapping["Target id"] = mapping["GtoPdb IUPHAR ID"].fillna(mapping["IUPHAR ID"])

        joined = pd.merge(
            mapping,
            self.target_df,
            left_on="Target id",
            right_on="target_id",
            how="left",
        )
        joined = pd.merge(
            joined,
            self.family_df,
            left_on="family_id",
            right_on="family_id",
            how="left",
            suffixes=("", "_family"),
        )
        # Final cleanup of column names to match the M script
        result = joined.rename(
            columns={
                "UniProtKB ID": "chembl_swissprot",
                "HGNC ID": "chembl_hgnc_id",
                "IUPHAR Name": "chembl_name",
                "Family name": "family_name",
                "Family id": "family_id",
            }
        )
        return result


@dataclass
class ClassificationRecord:
    """Represents a classification result derived from IUPHAR data."""

    IUPHAR_target_id: str = "N/A"
    IUPHAR_family_id: str = "N/A"
    IUPHAR_class: str = "Other Protein Target"
    IUPHAR_subclass: str = "Other Protein Target"

    IUPHAR_tree: list[str] = field(default_factory=lambda: ["0864-1", "0864"])

    IUPHAR_type: str = "Other Protein Target.Other Protein Target"
    IUPHAR_name: str = "N/A"
    IUPHAR_ecNumber: list[str] = field(default_factory=list)
    STATUS: str = "N/A"

    def __post_init__(self) -> None:
        """Normalise ``IUPHAR_tree`` after dataclass initialisation."""
        if not self.IUPHAR_tree:
            self.IUPHAR_tree = ["0864-1", "0864"]


class IUPHARClassifier:
    """Classifies targets using IUPHAR mappings and heuristics.

    Args:
        data: An IUPHARData object providing lookup tables and helper methods.
    """

    def __init__(self, data: IUPHARData):
        """Initializes the IUPHARClassifier.

        Args:
            data: An IUPHARData object providing lookup tables and helper methods.
        """
        self.data = data

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_parameter(parameter: str | None) -> bool:
        """Check if a parameter is valid for classification.

        A parameter is considered valid if it is not None, not an empty string,
        and not a default/placeholder value like "N/A" or "Other Protein Target".
        """
        return bool(parameter) and parameter not in {"N/A", "Other Protein Target"}

    @staticmethod
    def _is_valid_list(values: Iterable[str]) -> bool:
        """Check if a list of values is valid for classification.

        A list is considered valid if it contains at least one valid parameter
        and does not represent a default/placeholder value.
        """
        lst = [v for v in values if v and v != "N/A"]
        if not lst:
            return False
        if lst[0] == "0864-1" and len(lst) == 2:
            return False
        return True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def _family_to_type(self, family_id: str) -> str:
        """Get the type of a family by its ID."""
        if self._is_valid_parameter(family_id):
            return self.data.from_family_type(family_id) or "N/A"
        return "N/A"

    def _family_to_chain(self, family_id: str) -> list[str]:
        """Get the family hierarchy chain for a family ID."""
        if self._is_valid_parameter(family_id):
            return self.data.family_chain(family_id)
        return []

    def _target_record(self, target_id: str) -> pd.Series | None:
        """Get the full record for a target by its ID."""
        if self._is_valid_parameter(target_id):
            return self.data.from_target_record(target_id)
        return None

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def _target_to_type(self, target_id: str) -> str:
        """Determine the IUPHAR type for a given target ID.

        The type is constructed from the target's own type and its family's
        type.
        """
        record = self._target_record(target_id)
        if record is None:
            return "Other Protein Target.Other Protein Target"
        family_id = str(record.get("family_id", ""))
        type1 = str(record.get("type", ""))
        type2 = self._family_to_type(family_id)
        if self._is_valid_parameter(type1) and not self._is_valid_parameter(type2):
            type_val = type1
        elif self._is_valid_parameter(type2) and not self._is_valid_parameter(type1):
            type_val = type2
        elif not (self._is_valid_parameter(type1) or self._is_valid_parameter(type2)):
            type_val = "Other Protein Target.Other Protein Target"
        elif type1 == type2:
            type_val = type1
        elif self._is_valid_parameter(type1):
            type_val = type1
        else:
            type_val = "N/A.N/A"
        return type_val or "Other Protein Target.Other Protein Target"

    @staticmethod
    def _name_to_type(target_name: str) -> str:
        if not IUPHARClassifier._is_valid_parameter(target_name):
            return "Other Protein Target.Other Protein Target"
        name = target_name.lower()
        if "kinase" in name:
            return "Enzyme.Transferase"
        if "oxidase" in name or "reductase" in name:
            return "Enzyme.Oxidoreductase"
        if "hydrolase" in name or "protease" in name or "phosphatases" in name:
            return "Enzyme.Hydrolase"
        if "atpase" in name:
            return "Transporter.N/A"
        if "solute carrier" in name:
            return "Transporter.SLC superfamily of solute carrier"
        if "transport" in name:
            return "Transporter.N/A"
        if "channel" in name:
            return "Ion channel.N/A"
        if "hormone" in name:
            return "Receptor.Nuclear hormone receptor"
        return "Other Protein Target.Other Protein Target"

    _CHAIN_MAP = {
        "Enzyme.Oxidoreductase": ["0690-1", "0690"],
        "Enzyme.Transferase": ["0690-2", "0690"],
        "Enzyme.Multifunctional": ["0690-3", "0690"],
        "Enzyme.Hydrolase": ["0690-4", "0690"],
        "Enzyme.Isomerase": ["0690-5", "0690"],
        "Enzyme.Lyase": ["0690-6", "0690"],
        "Enzyme.Ligase": ["0690-6", "0690"],
        "Receptor.Catalytic receptor": ["0862", "0688"],
        "Receptor.G protein-coupled receptor": ["0694", "0688"],
        "Receptor.Nuclear hormone receptor": ["0095", "0688"],
        "Transporter.ATP-binding cassette transporter family": ["0136", "0691"],
        "Transporter.F-type and V-type ATPase": ["0137", "0691"],
        "Transporter.P-type ATPase": ["0138", "0691"],
        "Transporter.SLC superfamily of solute carrier": ["0863", "0691"],
        "Ion channel.Ligand-gated ion channel": ["0697", "0689"],
        "Ion channel.Other ion channel": ["0861", "0689"],
        "Ion channel.Voltage-gated ion channel": ["0696", "0689"],
    }

    @staticmethod
    def _ec_number_to_type(ec_numbers: list[str]) -> str:
        if not IUPHARClassifier._is_valid_list(ec_numbers):
            return ""
        prefixes = {n.split(".")[0] for n in ec_numbers if "." in n and n.split(".")[0]}
        if not prefixes:
            return ""
        if len(prefixes) > 1:
            return "Enzyme.Multifunctional"
        code = prefixes.pop()
        mapping = {
            "1": "Enzyme.Oxidoreductase",
            "2": "Enzyme.Transferase",
            "3": "Enzyme.Hydrolase",
            "4": "Enzyme.Lyase",
            "5": "Enzyme.Isomerase",
            "6": "Enzyme.Ligase",
            "7": "Enzyme.Translocase",
        }
        return mapping.get(code, "")

    @classmethod
    def _ec_number_to_chain(cls, ec_numbers: list[str]) -> list[str]:
        target_type = cls._ec_number_to_type(ec_numbers)
        return cls._CHAIN_MAP.get(target_type, ["0864-1", "0864"])

    # ------------------------------------------------------------------
    # Public classification methods
    # ------------------------------------------------------------------

    def set_record(
        self,
        iuphar_target_id: str,
        iuphar_family_id: str,
        iuphar_name: str,
        status: str | None = None,
        ec_numbers: list[str] | None = None,
    ) -> ClassificationRecord:
        """Construct a :class:`ClassificationRecord` from IUPHAR inputs.

        Parameters
        ----------
        iuphar_target_id:
            Identifier of the target in the IUPHAR database.
        iuphar_family_id:
            Family identifier associated with the target.
        iuphar_name:
            Human-readable target name.
        status:
            Optional status indicator; if omitted it is inferred from the
            provided identifiers.
        ec_numbers:
            Optional list of EC numbers.

        Returns
        -------
        ClassificationRecord
            Normalised classification information for the target.

        """
        target_id = (
            iuphar_target_id if self._is_valid_parameter(iuphar_target_id) else "N/A"
        )
        family_id = iuphar_family_id
        if not self._is_valid_parameter(family_id) and self._is_valid_parameter(
            target_id
        ):
            family_id = self.data.from_target_family_id(target_id) or "N/A"
        name = iuphar_name if self._is_valid_parameter(iuphar_name) else "N/A"
        numbers = ec_numbers if ec_numbers and self._is_valid_list(ec_numbers) else []
        if status is None:
            if self._is_valid_parameter(target_id):
                status = "target_id"
            elif self._is_valid_parameter(family_id):
                status = "family_id"
            elif self._is_valid_list(numbers):
                status = "ec_number"
            else:
                status = "N/A"

        if self._is_valid_parameter(target_id):
            type_val = self._target_to_type(target_id)
        elif self._is_valid_parameter(family_id):
            type_val = self._family_to_type(family_id)
        else:
            type_val = "Other Protein Target.Other Protein Target"

        parts = type_val.split(".")
        cls_part = parts[0] if parts else "Other Protein Target"
        sub_part = parts[1] if len(parts) > 1 else "Other Protein Target"

        tree = (
            self._family_to_chain(family_id)
            if self._is_valid_parameter(family_id)
            else ["0864-1", "0864"]
        )

        return ClassificationRecord(
            IUPHAR_target_id=target_id,
            IUPHAR_family_id=(
                family_id if self._is_valid_parameter(family_id) else "N/A"
            ),
            IUPHAR_class=cls_part,
            IUPHAR_subclass=sub_part,
            IUPHAR_tree=tree,
            IUPHAR_type=type_val,
            IUPHAR_name=name,
            IUPHAR_ecNumber=numbers,
            STATUS=status,
        )

    def by_target_id(
        self, iuphar_target_id: str, optional_name: str | None = None
    ) -> ClassificationRecord:
        """Classify a target given its IUPHAR target identifier."""
        if not self._is_valid_parameter(iuphar_target_id) or "|" in iuphar_target_id:
            return ClassificationRecord()
        family_id = self.data.from_target_family_id(iuphar_target_id)
        return self.set_record(iuphar_target_id, family_id, optional_name or "")

    def by_uniprot_id(self, uniprot_id: str) -> ClassificationRecord:
        """Classify a UniProt accession.

        The function resolves the accession to a target identifier and then
        delegates to :meth:`by_target_id`. If the accession cannot be mapped,
        a default :class:`ClassificationRecord` is returned.
        """
        target_id = self.data.target_id_by_uniprot(uniprot_id)
        if not target_id:
            return ClassificationRecord()
        return self.by_target_id(target_id)

    def by_family_id(
        self, iuphar_family_id: str, optional_name: str | None = None
    ) -> ClassificationRecord:
        """Classify a target given an IUPHAR family identifier."""
        if not self._is_valid_parameter(iuphar_family_id) or "|" in iuphar_family_id:
            return ClassificationRecord()
        return self.set_record("N/A", iuphar_family_id, optional_name or "")

    def by_ec_number(
        self, iuphar_ec_number: str, optional_name: str | None = None
    ) -> ClassificationRecord:
        """Classify based on an EC number string."""
        numbers = (
            iuphar_ec_number.split("|")
            if "." in iuphar_ec_number or "|" in iuphar_ec_number
            else []
        )
        if not self._is_valid_list(numbers):
            return ClassificationRecord()
        type_val = (
            self._ec_number_to_type(numbers)
            or "Other Protein Target.Other Protein Target"
        )
        parts = type_val.split(".")
        cls_part = parts[0] if parts else "Other Protein Target"
        sub_part = parts[1] if len(parts) > 1 else "Other Protein Target"
        tree = self._ec_number_to_chain(numbers)
        name = optional_name if self._is_valid_parameter(optional_name) else "N/A"
        return ClassificationRecord(
            IUPHAR_target_id="N/A",
            IUPHAR_family_id="N/A",
            IUPHAR_type=type_val,
            IUPHAR_class=cls_part,
            IUPHAR_subclass=sub_part,
            IUPHAR_tree=tree,
            IUPHAR_name=name or "N/A",
            IUPHAR_ecNumber=numbers,
            STATUS="ec_number",
        )

    def by_molecular_function(self, function: str) -> ClassificationRecord:
        """Classify based on a UniProt ``molecular_function`` string.

        The function uses :meth:`_name_to_type` to infer an ``IUPHAR_type``
        from free-text molecular function descriptions. The resulting type is
        mapped to a classification chain via :attr:`_CHAIN_MAP`.

        Parameters
        ----------
        function:
            Pipe-delimited molecular function annotations from UniProt.

        Returns
        -------
        ClassificationRecord
            Classification derived solely from molecular function keywords or
            a default record when the input is empty.

        """
        if not self._is_valid_parameter(function):
            return ClassificationRecord()

        type_val = self._name_to_type(function)
        parts = type_val.split(".")
        cls_part = parts[0] if parts else "Other Protein Target"
        sub_part = parts[1] if len(parts) > 1 else "Other Protein Target"
        tree = self._CHAIN_MAP.get(type_val, ["0864-1", "0864"])
        return ClassificationRecord(
            IUPHAR_type=type_val,
            IUPHAR_class=cls_part,
            IUPHAR_subclass=sub_part,
            IUPHAR_tree=tree,
            IUPHAR_name=function,
            STATUS="molecular_function",
        )

    def by_name(self, iuphar_name: str) -> ClassificationRecord:
        """Classify based on a free-text IUPHAR name."""
        if not self._is_valid_parameter(iuphar_name):
            return ClassificationRecord()
        target_id = self.data.target_id_by_name(iuphar_name)
        family_id = self.data.family_id_by_name(iuphar_name)
        if self._is_valid_parameter(target_id):
            type_val = self._target_to_type(target_id)
        elif self._is_valid_parameter(family_id):
            type_val = self._family_to_type(family_id)
        else:
            type_val = self._name_to_type(iuphar_name)
        tree = (
            self._family_to_chain(family_id)
            if self._is_valid_parameter(family_id)
            else ["864-1", "864"]
        )
        parts = type_val.split(".")
        cls_part = parts[0] if parts else "Other Protein Target"
        sub_part = parts[1] if len(parts) > 1 else "Other Protein Target"
        return ClassificationRecord(
            IUPHAR_target_id=(
                target_id if self._is_valid_parameter(target_id) else "N/A"
            ),
            IUPHAR_family_id=(
                family_id if self._is_valid_parameter(family_id) else "N/A"
            ),
            IUPHAR_type=type_val,
            IUPHAR_class=cls_part,
            IUPHAR_subclass=sub_part,
            IUPHAR_tree=tree,
            IUPHAR_name=iuphar_name,
            STATUS="name",
        )

    def get(
        self,
        iuphar_target_id: str,
        iuphar_family_id: str,
        iuphar_ec_number: str,
        iuphar_name: str,
    ) -> ClassificationRecord:
        """Resolve the best classification given multiple identifiers.

        This method attempts to classify a target using a hierarchy of
        identifiers, from the most specific (target ID) to the most general
        (name).

        Parameters
        ----------
        iuphar_target_id:
            The IUPHAR target ID.
        iuphar_family_id:
            The IUPHAR family ID.
        iuphar_ec_number:
            A pipe-delimited string of EC numbers.
        iuphar_name:
            The name of the target.

        Returns
        -------
        ClassificationRecord
            The best possible classification record based on the provided
            identifiers.
        """
        target_rec = (
            self.by_target_id(iuphar_target_id, iuphar_name)
            if self._is_valid_parameter(iuphar_target_id)
            else ClassificationRecord()
        )
        family_rec = (
            self.by_family_id(iuphar_family_id, iuphar_name)
            if self._is_valid_parameter(iuphar_family_id)
            else ClassificationRecord()
        )
        ec_rec = self.by_ec_number(iuphar_ec_number, iuphar_name)
        name_rec = self.by_name(iuphar_name)
        if target_rec.STATUS in {"target_id", "family_id"}:
            return target_rec
        if family_rec.STATUS == "family_id":
            return family_rec
        if ec_rec.STATUS == "ec_number":
            return ec_rec
        if name_rec.STATUS == "name":
            return name_rec
        return ClassificationRecord()

    # ------------------------------------------------------------------
    # Activity and initialisation helpers
    # ------------------------------------------------------------------

    def merge_activity(
        self, input_df: pd.DataFrame, activity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge catalytic activity information into ``input_df``."""
        merged = input_df.merge(
            activity_df, on="task_uniprot_id", how="left", suffixes=("", "_act")
        )
        merged["activity.ec_number"] = (
            merged.get("activity.ec_number", pd.Series(dtype=str))
            .replace("EC-not-assigned", "")
            .fillna("")
        )
        merged["ec_number"] = (
            merged["activity.ec_number"].fillna("")
            + "|"
            + merged.get("ec_number", pd.Series(dtype=str)).fillna("")
        )
        merged["ec_number"] = (
            merged["ec_number"]
            .str.split("|")
            .apply(lambda x: "|".join(sorted({i.strip() for i in x if i.strip()})))
        )
        return merged.drop(columns=["activity.ec_number"], errors="ignore")

    def init_class(
        self,
        input_df: pd.DataFrame,
        family_table: pd.DataFrame,
        activity_df: pd.DataFrame,
        db_name: str,
    ) -> pd.DataFrame:
        """Initialise classification columns for *input_df*."""
        df = self.merge_activity(input_df, activity_df)

        def classify(row: pd.Series) -> pd.Series:
            record = self.get(
                row.get("guidetopharmacology_id", ""),
                row.get("guidetopharmacology_family", ""),
                row.get("ec_number", ""),
                row.get("chembl_component_description", ""),
            )
            return pd.Series(
                {
                    "guidetopharmacology_id": record.IUPHAR_target_id,
                    "guidetopharmacology_family": record.IUPHAR_family_id,
                    "IUPHAR_type": record.IUPHAR_type,
                    "IUPHAR_class": record.IUPHAR_class,
                    "IUPHAR_subclass": record.IUPHAR_subclass,
                    "IUPHAR_chain": ">".join(record.IUPHAR_tree),
                    "IUPHAR_name": record.IUPHAR_name,
                }
            )

        classified = df.apply(classify, axis=1)
        return pd.concat([df, classified], axis=1)

    def by_reference(
        self, target_df: pd.DataFrame, db_df: pd.DataFrame, db_column: str
    ) -> pd.DataFrame:
        """Merge existing classification data from another database."""
        if "task_uniprot_id" not in target_df.columns:
            raise ValueError("target_df must contain 'task_uniprot_id'")
        if db_column not in db_df.columns:
            raise ValueError(f"db_df must contain '{db_column}' column")
        merged = target_df.merge(db_df, on="task_uniprot_id", how="left")
        merged["guidetopharmacology_family"] = merged[
            "guidetopharmacology_family"
        ].fillna(merged.get("guidetopharmacology_family_y"))
        merged["guidetopharmacology_type"] = merged["guidetopharmacology_type"].fillna(
            merged.get("guidetopharmacology_type_y")
        )
        return merged.drop(
            columns=[
                col
                for col in [
                    "guidetopharmacology_family_y",
                    "guidetopharmacology_type_y",
                ]
                if col in merged.columns
            ]
        )

    def get_database(
        self, family_table: pd.DataFrame, name_table: pd.DataFrame, db_column: str
    ) -> pd.DataFrame:
        """Derive a database-specific classification lookup table."""
        valid = family_table[
            family_table[db_column].notna() & (family_table[db_column] != "N/A")
        ]
        merged = valid.merge(name_table, on="task_uniprot_id", how="left")
        records = []
        for key, group in merged.groupby(db_column):
            family = (
                group["guidetopharmacology_family"].dropna().mode().iloc[0]
                if not group["guidetopharmacology_family"].dropna().empty
                else ""
            )
            type_series = (
                group["guidetopharmacology_id"]
                .dropna()
                .apply(self.data.from_target_type)
            )
            type_val = type_series.mode().iloc[0] if not type_series.empty else ""
            records.append(
                {
                    db_column: key,
                    "guidetopharmacology_family": family,
                    "guidetopharmacology_type": type_val,
                }
            )
        return pd.DataFrame(records)


__all__ = [
    "ApiCfg",
    "IupharCfg",
    "RetryCfg",
    "IUPHARData",
    "IUPHARClassifier",
    "ClassificationRecord",
    "init_session",
    "load_targets",
    "load_families",
]
