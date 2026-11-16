"""Quick checks against the hand-worked numbers in our report. Run: python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi

from hirag.data import load_mcpbench, make_tool_id
from hirag.fusion import _fuse, rrf_code, wrrf_paper
from hirag.metrics import ndcg_at_k, recall_at_k

TOY = [["search", "files", "name", "folder"], ["delete", "file", "folder"], ["search", "web", "bing"],
       ["read", "contents", "word", "docx", "file"], ["write", "text", "word", "docx", "file"],
       ["search", "code", "repository"]]
TOY_Q = ["read", "docx", "file", "search", "bing", "title"]


def test_bm25_toy_scores():
    s = BM25Okapi(TOY).get_scores(TOY_Q)
    assert round(s[3], 4) == 1.6598 and round(s[2], 4) == 1.4402 and round(s[4], 4) == 0.517


def test_wrrf_toy():
    bm25 = [3, 2, 4, 0, 1, 5]      # read_docx, bing_search, write_docx, file_search, file_delete, code_search
    dense = [5, 3, 4, 2, 0, 1]
    assert wrrf_paper(bm25, dense, 0.7)[:3] == [3, 2, 4]   # {Docx, Bing}
    assert wrrf_paper(bm25, dense, 0.1)[:3] == [5, 3, 4]   # Bing missed
    assert rrf_code([1, 2], [2, 1], 0.1)[0] == 2           # dense weight 1.0 dominates


def test_tool_id():
    assert make_tool_id("docx_mcp", "/read-docx") == "docx_mcp_read_docx"


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert ndcg_at_k(["a", "b"], ["a"], 2) == 1.0


def test_mcpbench_counts():
    b = load_mcpbench()
    assert len(b.tools) == 201 and len(b.services) == 39 and len(b.types) == 8
    assert len(b.single) == 201 and len(b.multi) == 60
