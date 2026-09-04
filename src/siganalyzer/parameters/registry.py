"""Central parameter registry and extensible extractor base class."""

from abc import ABC, abstractmethod
import importlib
import pkgutil
from typing import Type

from siganalyzer.common.logger import logger
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult


class BaseParameterExtractor(ABC):
    """Abstract base class for all signal parameter extraction plugins."""

    @property
    @abstractmethod
    def category(self) -> ParameterCategory:
        """Category handled by this extractor."""
        ...

    @abstractmethod
    def supported_parameters(self) -> list[ParameterDefinition]:
        """Definitions of all parameters extracted by this plugin."""
        ...

    @abstractmethod
    def extract(self, context: SignalContext) -> list[ParameterResult]:
        """Extract parameters from the given signal context.

        Must never raise an unhandled exception. If a value cannot be calculated,
        return a ParameterResult with status=ConfidenceLevel.UNKNOWN.
        """
        ...


class ParameterRegistry:
    """Central registry and coordinator of parameter extractors."""

    _extractors: list[Type[BaseParameterExtractor]] = []
    _instances: list[BaseParameterExtractor] | None = None

    @classmethod
    def register(cls, extractor_cls: Type[BaseParameterExtractor]) -> Type[BaseParameterExtractor]:
        """Decorator to register a parameter extractor class."""
        if extractor_cls not in cls._extractors:
            cls._extractors.append(extractor_cls)
            cls._instances = None  # Invalidate cached instances
            logger.debug(f"Registered extractor: {extractor_cls.__name__} for category {extractor_cls.category}")
        return extractor_cls

    @classmethod
    def get_extractors(cls) -> list[BaseParameterExtractor]:
        """Get instantiated extractors."""
        if cls._instances is None:
            # Ensure built-in extractors are loaded
            cls._ensure_extractors_loaded()
            cls._instances = [extractor_cls() for extractor_cls in cls._extractors]
        return cls._instances

    @classmethod
    def get_all_parameter_definitions(cls) -> list[ParameterDefinition]:
        """Retrieve schema definitions for all parameters registered across all extractors."""
        definitions: list[ParameterDefinition] = []
        for extractor in cls.get_extractors():
            definitions.extend(extractor.supported_parameters())
        return definitions

    @classmethod
    def _ensure_extractors_loaded(cls) -> None:
        """Dynamically discover and import all extractor modules in siganalyzer.parameters.extractors."""
        try:
            import siganalyzer.parameters.extractors as extractors_pkg
            for _, modname, _ in pkgutil.iter_modules(extractors_pkg.__path__):
                if modname.startswith("cat_"):
                    try:
                        importlib.import_module(f"siganalyzer.parameters.extractors.{modname}")
                    except Exception as e:
                        logger.error(f"Failed to load extractor module {modname}: {e}")
        except Exception as e:
            logger.debug(f"Extractor auto-discovery note: {e}")
