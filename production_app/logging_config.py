import logging as log


def setup_logging(log_level: str) -> None:
    """Configure application-wide logging."""

    log.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
