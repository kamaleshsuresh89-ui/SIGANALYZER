"""Parameter extraction orchestrator and SignalProfile container."""

import csv
from dataclasses import dataclass, field
import io
import json
from typing import Any

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.logger import logger
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterResult
from siganalyzer.parameters.registry import ParameterRegistry


@dataclass
class SignalProfile:
    """Comprehensive collection of all extracted parameters across Categories A through P."""

    results: list[ParameterResult] = field(default_factory=list)
    _by_id: dict[str, ParameterResult] = field(default_factory=dict, init=False)
    _by_category: dict[ParameterCategory, list[ParameterResult]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        """Index results by ID and Category for fast lookup."""
        self._by_id = {r.definition.id: r for r in self.results}
        self._by_category = {}
        for r in self.results:
            self._by_category.setdefault(r.definition.category, []).append(r)

    def get(self, param_id: str) -> ParameterResult | None:
        """Lookup a parameter by unique ID."""
        return self._by_id.get(param_id)

    def get_value(self, param_id: str, default: Any = None) -> Any:
        """Lookup a parameter value by ID."""
        res = self.get(param_id)
        if res is not None and res.value is not None and res.status != ConfidenceLevel.UNKNOWN:
            return res.value
        return default

    def get_category_results(self, category: ParameterCategory) -> list[ParameterResult]:
        """Get all parameters belonging to a specific category."""
        return self._by_category.get(category, [])

    def filter(
        self,
        category: ParameterCategory | None = None,
        status: ConfidenceLevel | None = None,
        search_query: str | None = None,
    ) -> list[ParameterResult]:
        """Multi-criteria search and filter across all parameters."""
        filtered = self.results
        if category is not None:
            filtered = [r for r in filtered if r.definition.category == category]
        if status is not None:
            filtered = [r for r in filtered if r.status == status]
        if search_query:
            q = search_query.strip().lower()
            filtered = [
                r for r in filtered
                if q in r.definition.name.lower()
                or q in r.definition.id.lower()
                or q in str(r.formatted_value).lower()
                or q in r.definition.unit.lower()
                or q in r.explanation.lower()
                or q in r.source.lower()
            ]
        return filtered

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def counts_by_status(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ConfidenceLevel}
        for r in self.results:
            counts[r.status.value] += 1
        return counts

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def to_csv(self) -> str:
        """Export all parameters to CSV format."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "Category", "ID", "Name", "Value", "Formatted Value", "Unit",
            "Status", "Confidence Score", "Source", "Method", "Explanation", "Validity Notes"
        ])
        for r in self.results:
            writer.writerow([
                r.definition.category.value,
                r.definition.id,
                r.definition.name,
                r.value,
                r.formatted_value,
                r.definition.unit,
                r.status.value,
                f"{r.confidence_score:.2f}",
                r.source,
                r.method,
                r.explanation,
                r.validity_notes,
            ])
        return out.getvalue()

    def to_json(self, indent: int = 2) -> str:
        """Export all parameters to JSON format."""
        return json.dumps(self.to_list_of_dicts(), indent=indent)


class ExtractionEngine:
    """Coordinates execution of all registered parameter extractors."""

    @classmethod
    def extract_all(cls, context: SignalContext) -> SignalProfile:
        """Execute all parameter extractors against the provided signal context."""
        extractors = ParameterRegistry.get_extractors()
        all_results: list[ParameterResult] = []

        for extractor in extractors:
            try:
                cat_results = extractor.extract(context)
                all_results.extend(cat_results)
            except Exception as e:
                logger.error(f"Extractor {extractor.__class__.__name__} encountered unhandled exception: {e}")
                # For each supported parameter, emit UNKNOWN result
                for defn in extractor.supported_parameters():
                    all_results.append(
                        ParameterResult(
                            definition=defn,
                            value=None,
                            status=ConfidenceLevel.UNKNOWN,
                            confidence_score=0.0,
                            source=extractor.__class__.__name__,
                            method="Failed",
                            explanation=f"Extraction aborted due to error: {e}",
                            validity_notes="Internal extractor error caught safely.",
                        )
                    )

        # Sort results by Category then export_priority
        all_results.sort(key=lambda r: (r.definition.category.value, r.definition.export_priority, r.definition.name))
        profile = SignalProfile(results=all_results)
        profile.rebuild_indexes()
        logger.info(f"Parameter extraction complete: {profile.total_count} parameters extracted across {len(profile._by_category)} categories.")
        return profile
