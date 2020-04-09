#!/usr/bin/env python3

# warnings.filterwarnings("ignore",
                        # category=FutureWarning)
import parser as parser
from analysis import main as analyze
from run import main as run
import sys
from utils import banner
import warnings

def main()->None:
    prs = parser.make_parser()
    args = prs.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'analyze':
        analyze(args)
    else:
        print('[ERROR] : Module {} does not exist.'\
              ' Choose one of the commands '\
              ' "run" and "analyze"'.format(args.command),
              )

if __name__ == "__main__":
    banner()
    warnings.filterwarnings("ignore",
                            category=FutureWarning)
    try:
        main()
    except KeyboardInterrupt:
        print("Terminated by user")
        sys.exit(-1)
