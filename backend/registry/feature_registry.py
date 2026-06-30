import logging
import traceback
import pandas as pd
from backend.registry.dynamic_registry import registry

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all registered indicators automatically.
    """

    feature_list = [df.copy()]

    success = 0
    failed = 0

    logger.info("Building %d indicators...", len(registry._registry))

    for key, func in registry._registry.items():

        try:
            result = func(df)

            if result is None:
                continue

            if isinstance(result, pd.Series):

                result = result.copy()

                if result.name is None:
                    result.name = key

                feature_list.append(result)
                success += 1

            elif isinstance(result, pd.DataFrame):

                result = result.copy()

                # Prefix only if required
                result.columns = [
                    f"{key}_{c}" if not str(c).startswith(f"{key}_") else str(c)
                    for c in result.columns
                ]

                feature_list.append(result)
                success += 1

            else:
                logger.warning(
                    "Skipped %s : returned %s",
                    key,
                    type(result).__name__
                )

        except Exception as e:

            failed += 1

            logger.error(
                "Indicator failed : %s",
                key
            )

            logger.debug(traceback.format_exc())

    final_df = pd.concat(feature_list, axis=1)

    # Remove duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Replace inf
    final_df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    final_df = final_df.ffill().bfill()

    logger.info(
        "Feature Build Completed | Success=%d | Failed=%d | Columns=%d",
        success,
        failed,
        len(final_df.columns)
    )

    return final_df
