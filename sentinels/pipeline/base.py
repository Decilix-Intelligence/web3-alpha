"""
Pipeline framework base classes.
Provides standardized task orchestration and execution flow.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineContext:
    """
    Pipeline context.
    Used to pass data and state between steps.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the context.

        Args:
            initial_data: initial data dictionary
        """
        self._data: Dict[str, Any] = initial_data or {}
        self._metadata: Dict[str, Any] = {
            "start_time": datetime.now(),
            "steps_completed": [],
            "errors": [],
        }

    def set(self, key: str, value: Any):
        """Set a value."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value."""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """Check whether a key exists."""
        return key in self._data

    def update(self, data: Dict[str, Any]):
        """Bulk-update data."""
        self._data.update(data)

    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all data."""
        return self._data.copy()

    def add_error(self, error: str):
        """Add an error message."""
        self._metadata["errors"].append({"time": datetime.now(), "message": error})

    def mark_step_completed(self, step_name: str):
        """Mark a step as completed."""
        self._metadata["steps_completed"].append({"name": step_name, "time": datetime.now()})

    def get_metadata(self) -> Dict[str, Any]:
        """Get a copy of the metadata."""
        return self._metadata.copy()


class PipelineStep(ABC):
    """
    Abstract base class for a pipeline step.
    Each concrete step must subclass this and implement execute().
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the step.

        Args:
            name: step name; uses the class name if None
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def execute(self, context: PipelineContext) -> bool:
        """
        Execute the step logic.

        Args:
            context: pipeline context

        Returns:
            True on success, False on failure
        """
        pass

    def on_error(self, context: PipelineContext, error: Exception):
        """
        Error-handling hook.

        Args:
            context: pipeline context
            error: exception object
        """
        error_msg = f"Step '{self.name}' failed: {str(error)}"
        self.logger.error(error_msg)
        context.add_error(error_msg)

    def should_skip(self, context: PipelineContext) -> bool:
        """
        Decide whether this step should be skipped.
        Subclasses can override this for conditional execution.

        Args:
            context: pipeline context

        Returns:
            True to skip, False to execute
        """
        return False


class Pipeline:
    """
    Pipeline runner.
    Executes a sequence of steps in order, with error handling and context passing.
    """

    def __init__(self, name: str, steps: List[PipelineStep], stop_on_error: bool = True):
        """
        Initialize the pipeline.

        Args:
            name: pipeline name
            steps: list of steps
            stop_on_error: whether to stop when an error occurs
        """
        self.name = name
        self.steps = steps
        self.stop_on_error = stop_on_error
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def run(self, context: Optional[PipelineContext] = None) -> PipelineContext:
        """
        Run the pipeline.

        Args:
            context: initial context; a new one is created if None

        Returns:
            The context object after execution
        """
        if context is None:
            context = PipelineContext()

        self.logger.info("=" * 70)
        self.logger.info(f"Starting Pipeline: {self.name}")
        self.logger.info(f"Total Steps: {len(self.steps)}")
        self.logger.info("=" * 70)

        for i, step in enumerate(self.steps, 1):
            # Check whether the step should be skipped
            if step.should_skip(context):
                self.logger.info(f"[{i}/{len(self.steps)}] Skipping step: {step.name}")
                continue

            self.logger.info(f"[{i}/{len(self.steps)}] Executing step: {step.name}")

            try:
                # Execute the step
                success = step.execute(context)

                if success:
                    context.mark_step_completed(step.name)
                    self.logger.info(f"  [OK] Step completed: {step.name}")
                else:
                    error_msg = f"Step returned False: {step.name}"
                    self.logger.error(f"  [FAIL] {error_msg}")
                    context.add_error(error_msg)

                    if self.stop_on_error:
                        self.logger.error(f"Pipeline stopped due to error in step: {step.name}")
                        break

            except Exception as e:
                # Call the error-handling hook
                step.on_error(context, e)

                if self.stop_on_error:
                    self.logger.error(f"Pipeline stopped due to exception in step: {step.name}")
                    raise
                else:
                    self.logger.warning(f"Continuing despite error in step: {step.name}")

        # Record the completion time
        metadata = context.get_metadata()
        duration = (datetime.now() - metadata["start_time"]).total_seconds()

        self.logger.info("=" * 70)
        self.logger.info(f"Pipeline Completed: {self.name}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Steps Completed: {len(metadata['steps_completed'])}/{len(self.steps)}")
        if metadata["errors"]:
            self.logger.warning(f"Errors: {len(metadata['errors'])}")
        self.logger.info("=" * 70)

        return context
