"""
main.py — Gold Bot Entry Point
Runs the complete pipeline in sequence or individual phases.

Usage:
  python main.py              → interactive menu
  python main.py --pipeline   → data fetch only
  python main.py --features   → feature engineering only
  python main.py --train      → model training only
  python main.py --backtest   → backtesting only
  python main.py --live       → start live bot
  python main.py --all        → run pipeline → features → train → backtest
"""
import sys
import argparse
from utils.logger import log


BANNER = """
╔══════════════════════════════════════════════════╗
║          GOLD BOT — XAU/USD AI TRADER            ║
║          Personal Edition | Demo Mode            ║
╚══════════════════════════════════════════════════╝
"""

MENU = """
Select what to run:

  [1] Data Pipeline     — Fetch MT5 OHLCV + Macro + COT data
  [2] Feature Engineering — Build technical + macro features
  [3] Train Models      — Walk-forward XGBoost + LightGBM training
  [4] Backtest          — Simulate trades on historical test data
  [5] Live Bot          — Start live signal + execution loop (DEMO)
  [6] Full Pipeline     — Run 1 → 2 → 3 → 4 in sequence

  [0] Exit
"""


def run_pipeline():
    log.info("Running: Data Pipeline")
    from data.pipeline import run
    run()


def run_features():
    log.info("Running: Feature Engineering")
    from data.features import run
    run()


def run_training():
    log.info("Running: Model Training")
    from models.train import run
    run()


def run_backtest():
    log.info("Running: Backtest Engine")
    from backtest.engine import run
    run()


def run_live():
    log.info("Running: Live Bot")
    from execution.live_bot import run
    run()


def run_all():
    log.info("Running full pipeline: Data → Features → Train → Backtest")
    run_pipeline()
    run_features()
    run_training()
    run_backtest()


def interactive_menu():
    print(BANNER)
    while True:
        print(MENU)
        choice = input("Enter choice: ").strip()
        if choice == "1":
            run_pipeline()
        elif choice == "2":
            run_features()
        elif choice == "3":
            run_training()
        elif choice == "4":
            run_backtest()
        elif choice == "5":
            run_live()
        elif choice == "6":
            run_all()
        elif choice == "0":
            log.info("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")


def main():
    parser = argparse.ArgumentParser(description="Gold Bot — XAU/USD AI Trader")
    parser.add_argument("--pipeline",  action="store_true", help="Run data pipeline")
    parser.add_argument("--features",  action="store_true", help="Run feature engineering")
    parser.add_argument("--train",     action="store_true", help="Train models")
    parser.add_argument("--backtest",  action="store_true", help="Run backtest")
    parser.add_argument("--live",      action="store_true", help="Start live bot")
    parser.add_argument("--all",       action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    print(BANNER)

    if args.pipeline:
        run_pipeline()
    elif args.features:
        run_features()
    elif args.train:
        run_training()
    elif args.backtest:
        run_backtest()
    elif args.live:
        run_live()
    elif args.all:
        run_all()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
