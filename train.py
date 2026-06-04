import sys

from train_experiment import main


if __name__ == "__main__":
    argv = [sys.argv[0]]
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "-e":
            argv.append("--max-epochs")
            idx += 1
            argv.append(args[idx])
        elif arg == "-r":
            argv.append("--seeds")
            idx += 1
            argv.append(args[idx])
        elif arg == "-mode":
            idx += 1
            if args[idx] != "dp_scl":
                raise SystemExit("This repository now supports only -mode dp_scl.")
        elif arg == "--contrastive":
            pass
        else:
            argv.append(arg)
        idx += 1
    sys.argv = argv
    main()
