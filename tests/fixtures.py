"""Reusable pytest fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture()
def pubmed_xml_factory() -> Callable[[Sequence[tuple[str, str, str | None]]], str]:
    """Return a helper that renders minimal PubMed XML documents."""

    def _build(records: Sequence[tuple[str, str, str | None]]) -> str:
        articles: list[str] = []
        for pmid, title, pub_type in records:
            publication_type = pub_type or "Journal Article"
            articles.append(
                """
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>{title}</ArticleTitle>
        <Abstract>
          <AbstractText>{title} abstract</AbstractText>
        </Abstract>
        <Journal>
          <Title>{title} Journal</Title>
        </Journal>
        <PublicationTypeList>
          <PublicationType>{publication_type}</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.{pmid}/doi{pmid}</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
""".strip().format(
                    pmid=pmid, title=title, publication_type=publication_type
                )
            )
        return (
            "<?xml version='1.0'?>\n<PubmedArticleSet>\n"
            + "\n".join(articles)
            + "\n</PubmedArticleSet>"
        )

    return _build


@pytest.fixture()
def metadata_csv(tmp_path: Path) -> Path:
    """Create a small CSV file suitable for metadata generation tests."""

    dataset = tmp_path / "dataset.csv"
    dataset.write_text("col\nvalue\n", encoding="utf-8")
    return dataset
