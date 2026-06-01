"""Spec-aligned scope selection over discovered PDF pages."""

from __future__ import annotations

from symbol_matching.models import PageRecord

SCOPE_THIS_PAGE = "this_page"
SCOPE_SIMILAR_NAME = "similar_page_name"
SCOPE_PAGE_TYPE = "same_page_type"
SCOPE_ALL_PAGES = "all_pages"

ALL_SCOPES = (SCOPE_THIS_PAGE, SCOPE_SIMILAR_NAME, SCOPE_PAGE_TYPE, SCOPE_ALL_PAGES)


def select_pages_for_scope(
    pages: list[PageRecord],
    reference_page_id: str,
    scope: str,
) -> list[PageRecord]:
    """Return the page subset that the matcher should search."""
    refs = [page for page in pages if page.id == reference_page_id]
    if len(refs) != 1:
        raise ValueError(f"reference page id not found exactly once: {reference_page_id}")
    ref = refs[0]

    if scope == SCOPE_THIS_PAGE:
        return [ref]
    if scope == SCOPE_ALL_PAGES:
        return list(pages)
    if scope == SCOPE_SIMILAR_NAME:
        family_matches = [page for page in pages if page.plan_family == ref.plan_family]
        if len(family_matches) == 0:
            return [ref]
        return family_matches
    if scope == SCOPE_PAGE_TYPE:
        if ref.page_type == "unknown":
            return [ref]
        return [page for page in pages if page.page_type == ref.page_type]
    raise ValueError(f"unsupported scope: {scope}")
