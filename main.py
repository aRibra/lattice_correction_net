# main.py - Main entry point for lattice correction network CLI
#
# This module now uses the new subcommand-based CLI. For usage examples:
#   python main.py --help
#   python main.py generate --help
#   python main.py train --help
#   python main.py evaluate --help
#   python main.py benchmark --help

from cli import main

if __name__ == '__main__':
    exit(main())
