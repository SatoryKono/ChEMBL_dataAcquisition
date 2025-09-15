"""Tests for the PubMed client."""

from __future__ import annotations

import sys
from pathlib import Path

import requests_mock

# Ensure modules are importable when the test is executed in isolation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.http_client import HttpClient  # type: ignore  # noqa: E402
from library import pubmed_client as pc  # type: ignore  # noqa: E402


SAMPLE_XML = """<?xml version='1.0'?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>1</PMID>
      <Article>
        <ArticleTitle>Title 1</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract 1</AbstractText>
        </Abstract>
        <Journal><Title>Journal 1</Title></Journal>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1/doi1</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>2</PMID>
      <Article>
        <ArticleTitle>Title 2</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract 2</AbstractText>
        </Abstract>
        <Journal><Title>Journal 2</Title></Journal>
        <PublicationTypeList>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.2/doi2</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_fetch_pubmed_records_parses_fields():
    with requests_mock.Mocker() as m:
        m.get(pc.API_URL, text=SAMPLE_XML)
        client = HttpClient(timeout=1.0, max_retries=1, rps=0)
        records = pc.fetch_pubmed_records(["1", "2"], client=client, batch_size=200)
    assert [r.pmid for r in records] == ["1", "2"]
    assert records[0].doi == "10.1/doi1"
    assert records[0].title == "Title 1"
    assert pc.classify_publication(records[0].publication_types) == "experimental"
    assert pc.classify_publication(records[1].publication_types) == "review"
