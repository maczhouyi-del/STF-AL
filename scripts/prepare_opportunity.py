from prepare_common import build_parser, prepare_dataset


if __name__ == "__main__":
    args = build_parser("opportunity").parse_args()
    prepare_dataset(args.config, "opportunity")
