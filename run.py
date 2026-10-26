import argparse

from production_app.main import main, predict_main


def cli() -> None:
    """Run training or prediction from the command line."""

    parser = argparse.ArgumentParser(description="Breast Cancer Classification CLI")

    parser.add_argument(
        "command",
        choices=["train", "predict"],
        help="Choose whether to train or predict.",
    )

    args = parser.parse_args()

    if args.command == "train":
        main()
    elif args.command == "predict":
        predict_main()


if __name__ == "__main__":
    cli()
