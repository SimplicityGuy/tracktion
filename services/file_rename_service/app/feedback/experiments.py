"""A/B testing experiment management for model versions."""

import logging
import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import stats

from services.file_rename_service.app.feedback.models import ABExperiment, ExperimentStatus, Feedback, FeedbackAction
from services.file_rename_service.app.feedback.storage import FeedbackStorage


class ExperimentError(Exception):
    """Base exception for experiment operations."""


class ExperimentValidationError(ExperimentError):
    """Exception for experiment validation failures."""


class ExperimentAllocationError(ExperimentError):
    """Exception for variant allocation failures."""


class StatisticalAnalysisError(ExperimentError):
    """Exception for statistical analysis failures."""


logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manage A/B testing experiments for model versions."""

    def __init__(
        self,
        storage: FeedbackStorage,
        min_sample_size: int = 100,
        significance_level: float = 0.05,
        power_threshold: float = 0.8,
    ):
        """Initialize experiment manager.

        Args:
            storage: Feedback storage backend
            min_sample_size: Minimum sample size per variant
            significance_level: Statistical significance level (alpha)
            power_threshold: Statistical power threshold
        """
        self.storage = storage
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self._active_experiments: dict[str, ABExperiment] = {}

    async def create_experiment(
        self,
        name: str,
        variant_a: str,
        variant_b: str,
        traffic_split: float = 0.5,
        description: str | None = None,
        duration_hours: int = 24,
    ) -> ABExperiment:
        """Create new A/B test experiment.

        Args:
            name: Experiment name
            variant_a: Control model version
            variant_b: Treatment model version
            traffic_split: Traffic percentage to variant B (0-1)
            description: Experiment description
            duration_hours: Planned experiment duration

        Returns:
            Created experiment

        Raises:
            ExperimentValidationError: If parameters are invalid
            ExperimentError: If experiment creation fails
        """
        try:
            # Validate input parameters
            if not name or not isinstance(name, str) or len(name.strip()) == 0:
                raise ExperimentValidationError("Experiment name must be a non-empty string")

            if not variant_a or not isinstance(variant_a, str):
                raise ExperimentValidationError("Variant A must be a non-empty string")

            if not variant_b or not isinstance(variant_b, str):
                raise ExperimentValidationError("Variant B must be a non-empty string")

            if variant_a == variant_b:
                raise ExperimentValidationError("Variant A and B must be different")

            # Validate traffic split
            if not isinstance(traffic_split, int | float):
                raise ExperimentValidationError(f"Traffic split must be numeric, got {type(traffic_split).__name__}")

            if not (0.0 <= traffic_split <= 1.0) or not np.isfinite(traffic_split):
                raise ExperimentValidationError(f"Invalid traffic split: {traffic_split} (must be between 0.0 and 1.0)")

            # Validate duration
            if not isinstance(duration_hours, int) or duration_hours <= 0:
                raise ExperimentValidationError(f"Duration must be a positive integer, got {duration_hours}")

            if duration_hours > 24 * 30:  # Max 30 days
                raise ExperimentValidationError(f"Duration too long: {duration_hours} hours (max 720 hours)")

            # Check for existing active experiments
            try:
                active = [exp for exp in self._active_experiments.values() if exp.is_active()]
                if active:
                    logger.warning(f"Creating experiment while {len(active)} experiments are active")
            except Exception as e:
                logger.warning(f"Failed to check for active experiments: {type(e).__name__}")

            # Create experiment with error handling
            try:
                experiment_id = str(uuid4())
                current_time = datetime.now(UTC)

                experiment = ABExperiment(
                    id=experiment_id,
                    name=name.strip(),
                    description=description.strip() if description else None,
                    variant_a=variant_a.strip(),
                    variant_b=variant_b.strip(),
                    traffic_split=float(traffic_split),
                    start_date=current_time,
                    end_date=current_time + timedelta(hours=duration_hours),
                    status=ExperimentStatus.PENDING,
                )
            except Exception as e:
                raise ExperimentError(f"Failed to create experiment object: {type(e).__name__}") from None

            # Store experiment with error handling
            try:
                self._active_experiments[experiment.id] = experiment
            except Exception as e:
                raise ExperimentError(f"Failed to store experiment: {type(e).__name__}") from None

            logger.info(
                f"Created experiment '{name}' - A: {variant_a} ({1 - traffic_split:.0%}) "
                f"vs B: {variant_b} ({traffic_split:.0%})"
            )

            return experiment

        except (ExperimentValidationError, ExperimentError):
            raise
        except Exception as e:
            raise ExperimentError(f"Unexpected error creating experiment: {type(e).__name__}") from None

    async def start_experiment(self, experiment_id: str) -> ABExperiment:
        """Start an experiment.

        Args:
            experiment_id: Experiment ID to start

        Returns:
            Started experiment

        Raises:
            ExperimentValidationError: If experiment not found or already running
            ExperimentError: If experiment start fails
        """
        try:
            # Validate experiment ID
            if not experiment_id or not isinstance(experiment_id, str):
                raise ExperimentValidationError("Experiment ID must be a non-empty string")

            # Get experiment with error handling
            try:
                experiment = self._active_experiments.get(experiment_id)
            except Exception as e:
                raise ExperimentError(f"Failed to retrieve experiment: {type(e).__name__}") from None

            if not experiment:
                raise ExperimentValidationError(f"Experiment {experiment_id} not found")

            # Validate experiment state
            if experiment.status != ExperimentStatus.PENDING:
                raise ExperimentValidationError(
                    f"Experiment {experiment_id} is not pending (status: {experiment.status})"
                )

            # Check if experiment has expired before starting
            current_time = datetime.now(UTC)
            if experiment.end_date and current_time >= experiment.end_date:
                raise ExperimentValidationError(
                    f"Experiment {experiment_id} has expired (ended: {experiment.end_date})"
                )

            # Start experiment with error handling
            try:
                experiment.status = ExperimentStatus.RUNNING
                experiment.start_date = current_time
            except Exception as e:
                raise ExperimentError(f"Failed to update experiment status: {type(e).__name__}") from None

            logger.info(f"Started experiment {experiment.name} ({experiment_id})")

            return experiment

        except (ExperimentValidationError, ExperimentError):
            raise
        except Exception as e:
            raise ExperimentError(f"Unexpected error starting experiment: {type(e).__name__}") from None

    async def allocate_variant(self, experiment_id: str | None = None) -> str | None:
        """Allocate user to experiment variant.

        Args:
            experiment_id: Specific experiment ID or None for active experiment

        Returns:
            Allocated model version or None if no active experiment

        Raises:
            ExperimentAllocationError: If allocation fails
        """
        try:
            experiment = None

            # Find experiment with error handling
            if experiment_id:
                if not isinstance(experiment_id, str):
                    raise ExperimentAllocationError("Experiment ID must be a string") from None

                try:
                    experiment = self._active_experiments.get(experiment_id)
                except Exception as e:
                    raise ExperimentAllocationError(
                        f"Failed to retrieve experiment {experiment_id}: {type(e).__name__}"
                    ) from None
            else:
                # Find active experiment with error handling
                try:
                    active = [exp for exp in self._active_experiments.values() if exp.is_active()]
                    experiment = active[0] if active else None

                    if len(active) > 1:
                        logger.warning(f"Multiple active experiments found ({len(active)}), using first one")

                except Exception as e:
                    logger.error(f"Failed to find active experiments: {type(e).__name__}")
                    return None

            # Check if experiment exists and is active
            if not experiment:
                logger.debug(f"No {'active' if not experiment_id else 'matching'} experiment found")
                return None

            if not experiment.is_active():
                logger.debug(f"Experiment {experiment.id} is not active (status: {experiment.status})")
                return None

            # Allocate based on traffic split with error handling
            try:
                random_value = random.random()
                if not (0.0 <= random_value <= 1.0):
                    logger.warning(f"Invalid random value: {random_value}, regenerating")
                    random_value = 0.5  # Fallback value

                variant: str = experiment.allocate_variant(random_value)

                if variant not in [experiment.variant_a, experiment.variant_b]:
                    raise ExperimentAllocationError(
                        f"Invalid variant allocation: {variant} not in [{experiment.variant_a}, {experiment.variant_b}]"
                    )

            except Exception as e:
                raise ExperimentAllocationError(f"Failed to allocate variant: {type(e).__name__}") from None

            # Update sample counts with bounds checking
            try:
                if variant == experiment.variant_a:
                    experiment.sample_size_a += 1
                    # Prevent overflow (though unlikely in practice)
                    if experiment.sample_size_a < 0:
                        logger.warning("Sample size A overflow, resetting to 1")
                        experiment.sample_size_a = 1
                else:
                    experiment.sample_size_b += 1
                    if experiment.sample_size_b < 0:
                        logger.warning("Sample size B overflow, resetting to 1")
                        experiment.sample_size_b = 1

            except Exception as e:
                # Don't fail allocation due to sample count update issues
                logger.error(f"Failed to update sample counts: {type(e).__name__}")

            logger.debug(f"Allocated variant {variant} for experiment {experiment.id}")
            return variant

        except ExperimentAllocationError:
            raise
        except Exception as e:
            raise ExperimentAllocationError(f"Unexpected error during allocation: {type(e).__name__}") from None

    async def record_feedback(
        self,
        experiment_id: str,
        variant: str,
        feedback: Feedback,
    ) -> None:
        """Record feedback for experiment variant.

        Args:
            experiment_id: Experiment ID
            variant: Variant that generated the proposal
            feedback: User feedback

        Raises:
            ExperimentValidationError: If input parameters are invalid
        """
        try:
            # Validate inputs
            if not experiment_id or not isinstance(experiment_id, str):
                raise ExperimentValidationError("Experiment ID must be a non-empty string")

            if not variant or not isinstance(variant, str):
                raise ExperimentValidationError("Variant must be a non-empty string")

            if not feedback:
                raise ExperimentValidationError("Feedback cannot be None")

            # Get experiment with error handling
            try:
                experiment = self._active_experiments.get(experiment_id)
            except Exception as e:
                logger.error(f"Failed to retrieve experiment {experiment_id}: {type(e).__name__}")
                return

            if not experiment:
                logger.warning(f"Experiment {experiment_id} not found for feedback recording")
                return

            # Validate variant matches experiment
            if variant not in [experiment.variant_a, experiment.variant_b]:
                logger.warning(
                    f"Variant {variant} not valid for experiment {experiment_id} "
                    f"(expected {experiment.variant_a} or {experiment.variant_b})"
                )
                return

            # Determine which variant metrics to update
            try:
                metrics = experiment.metrics_a if variant == experiment.variant_a else experiment.metrics_b
            except Exception as e:
                logger.error(f"Failed to access metrics for variant {variant}: {type(e).__name__}")
                return

            # Initialize metrics if needed with error handling
            try:
                if "total" not in metrics:
                    metrics["total"] = 0
                    metrics["approved"] = 0
                    metrics["rejected"] = 0
                    metrics["modified"] = 0

                # Validate existing metrics
                for key in ["total", "approved", "rejected", "modified"]:
                    if key not in metrics or not isinstance(metrics[key], int | float) or metrics[key] < 0:
                        logger.warning(f"Invalid or missing metric {key}, resetting to 0")
                        metrics[key] = 0

            except Exception as e:
                logger.error(f"Failed to initialize metrics: {type(e).__name__}")
                return

            # Update metrics with validation
            try:
                metrics["total"] += 1

                # Validate feedback action
                if not hasattr(feedback, "user_action") or feedback.user_action is None:
                    logger.warning("Feedback missing user action, treating as rejected")
                    metrics["rejected"] += 1
                elif feedback.user_action == FeedbackAction.APPROVED:
                    metrics["approved"] += 1
                elif feedback.user_action == FeedbackAction.REJECTED:
                    metrics["rejected"] += 1
                elif feedback.user_action == FeedbackAction.MODIFIED:
                    metrics["modified"] += 1
                else:
                    logger.warning(f"Unknown feedback action: {feedback.user_action}, treating as rejected")
                    metrics["rejected"] += 1

                # Calculate rates with safe division
                total = metrics["total"]
                if total > 0:
                    metrics["approval_rate"] = metrics["approved"] / total
                    metrics["rejection_rate"] = metrics["rejected"] / total
                    metrics["modification_rate"] = metrics["modified"] / total
                else:
                    logger.warning("Total count is zero, cannot calculate rates")
                    metrics["approval_rate"] = 0.0
                    metrics["rejection_rate"] = 0.0
                    metrics["modification_rate"] = 0.0

                # Validate calculated rates
                for rate_key in ["approval_rate", "rejection_rate", "modification_rate"]:
                    rate_value = metrics[rate_key]
                    if not (0.0 <= rate_value <= 1.0) or not np.isfinite(rate_value):
                        logger.warning(f"Invalid {rate_key}: {rate_value}, setting to 0.0")
                        metrics[rate_key] = 0.0

            except Exception as e:
                logger.error(f"Failed to update metrics: {type(e).__name__}")
                return

            # Check if experiment should conclude with error handling
            try:
                if await self._should_conclude_experiment(experiment):
                    await self.conclude_experiment(experiment_id)
            except Exception as e:
                logger.error(f"Error checking experiment conclusion: {type(e).__name__}")
                # Don't return - feedback was still recorded successfully

        except ExperimentValidationError as e:
            logger.error(f"Validation error recording feedback: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error recording feedback: {type(e).__name__}")

    async def _should_conclude_experiment(self, experiment: ABExperiment) -> bool:
        """Check if experiment should be concluded.

        Args:
            experiment: Experiment to check

        Returns:
            True if experiment should conclude
        """
        try:
            # Validate experiment
            if not experiment:
                logger.warning("Cannot check conclusion criteria for None experiment")
                return False

            # Check if already concluded
            if experiment.status in [ExperimentStatus.CONCLUDED, ExperimentStatus.CANCELLED]:
                return False

            # Check end date with error handling
            try:
                if experiment.end_date:
                    current_time = datetime.now(UTC)
                    if current_time >= experiment.end_date:
                        logger.debug(f"Experiment {experiment.id} reached end date")
                        return True
            except Exception as e:
                logger.warning(f"Failed to check end date: {type(e).__name__}")

            # Check minimum sample size with validation
            try:
                sample_a = experiment.sample_size_a
                sample_b = experiment.sample_size_b

                if not isinstance(sample_a, int | float) or not isinstance(sample_b, int | float):
                    logger.warning(f"Invalid sample sizes: A={sample_a}, B={sample_b}")
                    return False

                if sample_a < 0 or sample_b < 0:
                    logger.warning(f"Negative sample sizes: A={sample_a}, B={sample_b}")
                    return False

                if sample_a < self.min_sample_size or sample_b < self.min_sample_size:
                    logger.debug(f"Insufficient samples: A={sample_a}, B={sample_b} (need {self.min_sample_size} each)")
                    return False

            except Exception as e:
                logger.warning(f"Failed to check sample sizes: {type(e).__name__}")
                return False

            # Check for statistical significance with error handling
            try:
                p_value = self._calculate_significance(experiment)
                if p_value is not None and p_value < self.significance_level:
                    # Check statistical power
                    try:
                        power = self._calculate_power(experiment)
                        if power is not None and power >= self.power_threshold:
                            logger.debug(
                                f"Experiment {experiment.id} meets statistical criteria: "
                                f"p={p_value:.4f} < {self.significance_level}, "
                                f"power={power:.3f} >= {self.power_threshold}"
                            )
                            return True
                        logger.debug(
                            f"Experiment {experiment.id} significant but underpowered: p={p_value:.4f}, power={power}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to calculate power for conclusion check: {type(e).__name__}")
                else:
                    logger.debug(f"Experiment {experiment.id} not significant: p={p_value}")
            except Exception as e:
                logger.warning(f"Failed to calculate significance for conclusion check: {type(e).__name__}")

            return False

        except Exception as e:
            logger.error(f"Unexpected error checking experiment conclusion: {type(e).__name__}")
            return False

    def _calculate_significance(self, experiment: ABExperiment) -> float | None:
        """Calculate statistical significance between variants.

        Args:
            experiment: Experiment to analyze

        Returns:
            P-value or None if insufficient data

        Raises:
            StatisticalAnalysisError: If statistical analysis fails
        """
        try:
            # Validate experiment
            if not experiment:
                raise StatisticalAnalysisError("Experiment cannot be None") from None

            # Get metrics with validation
            try:
                metrics_a = experiment.metrics_a
                metrics_b = experiment.metrics_b
            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to access experiment metrics: {type(e).__name__}") from None

            if not metrics_a or not metrics_b:
                logger.debug("Insufficient metrics data for significance calculation")
                return None

            # Check minimum sample sizes
            total_a = metrics_a.get("total", 0)
            total_b = metrics_b.get("total", 0)

            if not isinstance(total_a, int | float) or not isinstance(total_b, int | float):
                logger.warning("Invalid total counts in metrics")
                return None

            if total_a < 30 or total_b < 30:
                logger.debug(f"Sample sizes too small for significance test: A={total_a}, B={total_b} (need ≥30 each)")
                return None

            # Get approval counts with validation
            approved_a = metrics_a.get("approved", 0)
            approved_b = metrics_b.get("approved", 0)

            if not isinstance(approved_a, int | float) or not isinstance(approved_b, int | float):
                logger.warning("Invalid approved counts in metrics")
                return None

            # Validate data consistency
            if approved_a < 0 or approved_b < 0:
                logger.warning(f"Negative approved counts: A={approved_a}, B={approved_b}")
                return None

            if approved_a > total_a or approved_b > total_b:
                logger.warning(f"Approved counts exceed totals: A={approved_a}/{total_a}, B={approved_b}/{total_b}")
                return None

            # Create contingency table with validation
            try:
                not_approved_a = total_a - approved_a
                not_approved_b = total_b - approved_b

                if not_approved_a < 0 or not_approved_b < 0:
                    logger.warning(f"Negative rejection counts: A={not_approved_a}, B={not_approved_b}")
                    return None

                contingency = [
                    [int(approved_a), int(not_approved_a)],
                    [int(approved_b), int(not_approved_b)],
                ]

                # Validate contingency table
                for row in contingency:
                    for cell in row:
                        if cell < 0:
                            raise StatisticalAnalysisError(
                                f"Negative cell value in contingency table: {cell}"
                            ) from None

                # Check for minimum expected frequencies (chi-square assumption)
                row_totals = [sum(row) for row in contingency]
                col_totals = [sum(col) for col in zip(*contingency, strict=False)]
                grand_total = sum(row_totals)

                if grand_total == 0:
                    logger.warning("Empty contingency table")
                    return None

                # Calculate expected frequencies
                min_expected = float("inf")
                for i in range(2):
                    for j in range(2):
                        expected = (row_totals[i] * col_totals[j]) / grand_total
                        min_expected = min(min_expected, expected)

                if min_expected < 5:
                    logger.debug(f"Minimum expected frequency too low: {min_expected:.2f} < 5")
                    # Could use Fisher's exact test here instead, but chi-square is more common
                    # For now, continue with chi-square but log the limitation

            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to create contingency table: {type(e).__name__}") from None

            # Perform chi-square test
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

                # Validate results
                if not np.isfinite(chi2) or not np.isfinite(p_value):
                    raise StatisticalAnalysisError(
                        f"Invalid chi-square results: chi2={chi2}, p_value={p_value}"
                    ) from None

                if not (0.0 <= p_value <= 1.0):
                    raise StatisticalAnalysisError(f"Invalid p-value: {p_value} (must be between 0 and 1)") from None

                logger.debug(f"Significance test: chi2={chi2:.4f}, p_value={p_value:.4f}, dof={dof}")
                return float(p_value)

            except Exception as e:
                if "chi2_contingency" in str(e) or "scipy" in str(e).lower():
                    raise StatisticalAnalysisError(f"Chi-square test failed: {type(e).__name__}") from None
                raise

        except StatisticalAnalysisError:
            raise
        except Exception as e:
            raise StatisticalAnalysisError(f"Unexpected error calculating significance: {type(e).__name__}") from None

    def _calculate_power(self, experiment: ABExperiment) -> float | None:
        """Calculate statistical power of experiment.

        Args:
            experiment: Experiment to analyze

        Returns:
            Statistical power or None if insufficient data

        Raises:
            StatisticalAnalysisError: If power calculation fails
        """
        try:
            # Validate experiment
            if not experiment:
                raise StatisticalAnalysisError("Experiment cannot be None") from None

            # Get metrics with validation
            try:
                metrics_a = experiment.metrics_a
                metrics_b = experiment.metrics_b
            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to access experiment metrics: {type(e).__name__}") from None

            if not metrics_a or not metrics_b:
                logger.debug("Insufficient metrics data for power calculation")
                return None

            # Get approval rates with validation
            try:
                p1 = metrics_a.get("approval_rate", 0)
                p2 = metrics_b.get("approval_rate", 0)

                if not isinstance(p1, int | float) or not isinstance(p2, int | float):
                    logger.warning("Invalid approval rates in metrics")
                    return None

                # Validate rates are in valid range
                if not (0.0 <= p1 <= 1.0) or not (0.0 <= p2 <= 1.0):
                    logger.warning(f"Approval rates out of range: p1={p1}, p2={p2}")
                    return None

                if not np.isfinite(p1) or not np.isfinite(p2):
                    logger.warning(f"Non-finite approval rates: p1={p1}, p2={p2}")
                    return None

            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to get approval rates: {type(e).__name__}") from None

            # Calculate effect size (Cohen's h for proportions) with bounds checking
            try:
                # Handle edge cases for arcsin(sqrt(p)) calculation
                def safe_arcsin_sqrt(p: float) -> float:
                    if p <= 0:
                        return 0.0
                    if p >= 1:
                        return float(np.pi / 2)  # arcsin(1) = π/2
                    return float(np.arcsin(np.sqrt(p)))

                arcsin_p1 = safe_arcsin_sqrt(p1)
                arcsin_p2 = safe_arcsin_sqrt(p2)

                if not np.isfinite(arcsin_p1) or not np.isfinite(arcsin_p2):
                    logger.warning(f"Non-finite arcsin values: arcsin_p1={arcsin_p1}, arcsin_p2={arcsin_p2}")
                    return None

                # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
                h = 2 * (arcsin_p1 - arcsin_p2)

                if not np.isfinite(h):
                    logger.warning(f"Non-finite effect size: h={h}")
                    return None

            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to calculate effect size: {type(e).__name__}") from None

            # Get sample sizes with validation
            try:
                n_a = experiment.sample_size_a
                n_b = experiment.sample_size_b

                if not isinstance(n_a, int | float) or not isinstance(n_b, int | float):
                    logger.warning(f"Invalid sample sizes: n_a={n_a}, n_b={n_b}")
                    return None

                if n_a < 0 or n_b < 0:
                    logger.warning(f"Negative sample sizes: n_a={n_a}, n_b={n_b}")
                    return None

                n = min(n_a, n_b)
                if n == 0:
                    logger.debug("Zero sample size, cannot calculate power")
                    return None

            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to get sample sizes: {type(e).__name__}") from None

            # Simplified power calculation with bounds checking
            # In practice, use statsmodels or similar for accurate power analysis
            try:
                if self.min_sample_size <= 0:
                    logger.warning(f"Invalid minimum sample size: {self.min_sample_size}")
                    return None

                abs_h = abs(h)

                # Approximate power based on effect size and sample size
                if abs_h < 0.2:  # Small effect
                    power = min(0.2 * (n / self.min_sample_size), 0.8)
                elif abs_h < 0.5:  # Medium effect
                    power = min(0.5 * (n / self.min_sample_size), 0.95)
                else:  # Large effect
                    power = min(0.8 * (n / self.min_sample_size), 0.99)

                # Final validation
                if not np.isfinite(power) or not (0.0 <= power <= 1.0):
                    logger.warning(f"Invalid power calculation: {power}")
                    return None

                logger.debug(f"Power calculation: h={h:.4f}, n={n}, power={power:.4f}")
                return float(power)

            except Exception as e:
                raise StatisticalAnalysisError(f"Failed to calculate power: {type(e).__name__}") from None

        except StatisticalAnalysisError:
            raise
        except Exception as e:
            raise StatisticalAnalysisError(f"Unexpected error calculating power: {type(e).__name__}") from None

    async def conclude_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Conclude an experiment and determine winner.

        Args:
            experiment_id: Experiment ID to conclude

        Returns:
            Experiment results

        Raises:
            ExperimentValidationError: If experiment not found or cannot be concluded
            ExperimentError: If conclusion process fails
        """
        try:
            # Validate experiment ID
            if not experiment_id or not isinstance(experiment_id, str):
                raise ExperimentValidationError("Experiment ID must be a non-empty string")

            # Get experiment with error handling
            try:
                experiment = self._active_experiments.get(experiment_id)
            except Exception as e:
                raise ExperimentError(f"Failed to retrieve experiment: {type(e).__name__}") from None

            if not experiment:
                raise ExperimentValidationError(f"Experiment {experiment_id} not found")

            # Validate experiment can be concluded
            if experiment.status == ExperimentStatus.CONCLUDED:
                logger.warning(f"Experiment {experiment_id} already concluded")
                # Return existing results if available

            # Calculate final statistics with error handling
            p_value = None
            power = None

            try:
                p_value = self._calculate_significance(experiment)
            except StatisticalAnalysisError as e:
                logger.error(f"Failed to calculate significance: {e}")
            except Exception as e:
                logger.error(f"Unexpected error calculating significance: {type(e).__name__}")

            try:
                power = self._calculate_power(experiment)
            except StatisticalAnalysisError as e:
                logger.error(f"Failed to calculate power: {e}")
            except Exception as e:
                logger.error(f"Unexpected error calculating power: {type(e).__name__}")

            # Update experiment status with error handling
            try:
                experiment.statistical_significance = p_value
                experiment.status = ExperimentStatus.CONCLUDED
            except Exception as e:
                raise ExperimentError(f"Failed to update experiment status: {type(e).__name__}") from None

            # Determine winner with comprehensive error handling
            winner = None
            improvement = 0.0

            try:
                metrics_a = experiment.metrics_a or {}
                metrics_b = experiment.metrics_b or {}

                if p_value is not None and p_value < self.significance_level:
                    # Statistically significant difference
                    approval_a = metrics_a.get("approval_rate", 0)
                    approval_b = metrics_b.get("approval_rate", 0)

                    # Validate approval rates
                    if not isinstance(approval_a, int | float) or not isinstance(approval_b, int | float):
                        logger.warning("Invalid approval rates, cannot determine winner")
                    elif not (0.0 <= approval_a <= 1.0) or not (0.0 <= approval_b <= 1.0):
                        logger.warning(f"Approval rates out of range: A={approval_a}, B={approval_b}")
                    elif not np.isfinite(approval_a) or not np.isfinite(approval_b):
                        logger.warning(f"Non-finite approval rates: A={approval_a}, B={approval_b}")
                    else:
                        # Determine winner
                        if approval_b > approval_a:
                            winner = experiment.variant_b
                            improvement = (approval_b - approval_a) / approval_a if approval_a > 0 else float("inf")
                        elif approval_a > approval_b:
                            winner = experiment.variant_a
                            improvement = (approval_a - approval_b) / approval_b if approval_b > 0 else float("inf")
                        else:
                            # Exactly equal rates - no winner
                            winner = None
                            improvement = 0.0

                        # Cap improvement at reasonable values
                        if improvement == float("inf"):
                            improvement = 10.0  # 1000% improvement cap
                        elif improvement > 10.0:
                            improvement = 10.0
                        elif not np.isfinite(improvement):
                            improvement = 0.0
                else:
                    # No significant difference
                    winner = None
                    improvement = 0.0

                experiment.winner = winner

            except Exception as e:
                logger.error(f"Failed to determine winner: {type(e).__name__}")
                winner = None
                improvement = 0.0
                experiment.winner = None

            # Calculate duration with error handling
            try:
                if experiment.start_date:
                    duration_hours = (datetime.now(UTC) - experiment.start_date).total_seconds() / 3600
                    duration_hours = max(0.0, duration_hours)  # Ensure non-negative
                else:
                    duration_hours = 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate duration: {type(e).__name__}")
                duration_hours = 0.0

            # Build results with validation
            try:
                results = {
                    "experiment_id": experiment_id,
                    "name": experiment.name,
                    "winner": winner,
                    "statistical_significance": p_value,
                    "statistical_power": power,
                    "improvement_percentage": improvement * 100 if np.isfinite(improvement) else 0.0,
                    "variant_a": {
                        "version": experiment.variant_a,
                        "sample_size": experiment.sample_size_a,
                        "metrics": experiment.metrics_a or {},
                    },
                    "variant_b": {
                        "version": experiment.variant_b,
                        "sample_size": experiment.sample_size_b,
                        "metrics": experiment.metrics_b or {},
                    },
                    "duration_hours": duration_hours,
                }
            except Exception as e:
                raise ExperimentError(f"Failed to build results: {type(e).__name__}") from None

            # Log results with error handling
            try:
                winner_str = winner or "No significant difference"
                p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                power_str = f"{power:.2f}" if power is not None else "N/A"

                logger.info(
                    f"Concluded experiment '{experiment.name}' - Winner: {winner_str} (p={p_str}, power={power_str})"
                )
            except Exception as e:
                logger.warning(f"Failed to log conclusion: {type(e).__name__}")

            return results

        except (ExperimentValidationError, ExperimentError):
            raise
        except Exception as e:
            raise ExperimentError(f"Unexpected error concluding experiment: {type(e).__name__}") from None

    async def get_active_experiments(self) -> list[ABExperiment]:
        """Get list of active experiments.

        Returns:
            List of active experiments
        """
        return [exp for exp in self._active_experiments.values() if exp.is_active()]

    async def get_experiment_status(self, experiment_id: str) -> dict[str, Any] | None:
        """Get detailed experiment status.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment status or None if not found
        """
        try:
            # Validate experiment ID
            if not experiment_id or not isinstance(experiment_id, str):
                logger.warning("Invalid experiment ID for status retrieval")
                return None

            # Get experiment with error handling
            try:
                experiment = self._active_experiments.get(experiment_id)
            except Exception as e:
                logger.error(f"Failed to retrieve experiment {experiment_id}: {type(e).__name__}")
                return None

            if not experiment:
                return None

            # Calculate statistics with error handling
            p_value = None
            power = None

            try:
                p_value = self._calculate_significance(experiment)
            except Exception as e:
                logger.warning(f"Failed to calculate significance for status: {type(e).__name__}")

            try:
                power = self._calculate_power(experiment)
            except Exception as e:
                logger.warning(f"Failed to calculate power for status: {type(e).__name__}")

            # Determine if experiment can be concluded
            can_conclude = False
            try:
                if p_value is not None and power is not None:
                    can_conclude = p_value < self.significance_level and power >= self.power_threshold
            except Exception as e:
                logger.warning(f"Failed to determine conclusion readiness: {type(e).__name__}")

            # Build status response with error handling
            try:
                return {
                    "id": experiment.id,
                    "name": experiment.name,
                    "status": experiment.status.value,
                    "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
                    "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                    "variant_a": {
                        "version": experiment.variant_a,
                        "sample_size": experiment.sample_size_a,
                        "metrics": experiment.metrics_a or {},
                    },
                    "variant_b": {
                        "version": experiment.variant_b,
                        "sample_size": experiment.sample_size_b,
                        "metrics": experiment.metrics_b or {},
                    },
                    "traffic_split": experiment.traffic_split,
                    "statistical_significance": p_value,
                    "statistical_power": power,
                    "can_conclude": can_conclude,
                }

            except Exception as e:
                logger.error(f"Failed to build status response: {type(e).__name__}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error getting experiment status: {type(e).__name__}")
            return None

    async def cancel_experiment(self, experiment_id: str) -> ABExperiment:
        """Cancel an experiment.

        Args:
            experiment_id: Experiment ID to cancel

        Returns:
            Cancelled experiment

        Raises:
            ExperimentValidationError: If experiment not found
            ExperimentError: If cancellation fails
        """
        try:
            # Validate experiment ID
            if not experiment_id or not isinstance(experiment_id, str):
                raise ExperimentValidationError("Experiment ID must be a non-empty string")

            # Get experiment with error handling
            try:
                experiment = self._active_experiments.get(experiment_id)
            except Exception as e:
                raise ExperimentError(f"Failed to retrieve experiment: {type(e).__name__}") from None

            if not experiment:
                raise ExperimentValidationError(f"Experiment {experiment_id} not found")

            # Check if already cancelled/concluded
            if experiment.status in [ExperimentStatus.CANCELLED, ExperimentStatus.CONCLUDED]:
                logger.warning(f"Experiment {experiment_id} already in final state: {experiment.status}")
                return experiment

            # Cancel experiment with error handling
            try:
                experiment.status = ExperimentStatus.CANCELLED
                experiment.end_date = datetime.now(UTC)
            except Exception as e:
                raise ExperimentError(f"Failed to update experiment status: {type(e).__name__}") from None

            logger.info(f"Cancelled experiment '{experiment.name}' ({experiment_id})")

            return experiment

        except (ExperimentValidationError, ExperimentError):
            raise
        except Exception as e:
            raise ExperimentError(f"Unexpected error cancelling experiment: {type(e).__name__}") from None
