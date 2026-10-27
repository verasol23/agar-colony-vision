"""Command line entry point for reproducible dataset processing."""
import argparse
from dataclasses import replace
from .batch import run_batch
from .config import DetectorConfig, load_preset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the Improved AGAR colony detector')
    parser.add_argument('--dataset', required=True, help='Folder with JPG/JSON pairs')
    parser.add_argument('--output', required=True, help='Separate result folder')
    parser.add_argument('--limit', type=int, default=None, help='Optional number of images')
    parser.add_argument('--preset', help='JSON parameter preset saved from GUI')
    parser.add_argument('--iou', type=float, default=None)
    parser.add_argument('--dish-shrink', type=float, default=None)
    parser.add_argument('--no-images', action='store_true')
    parser.add_argument('--save-masks', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_preset(args.preset) if args.preset else DetectorConfig()
    overrides = {}
    if args.iou is not None:
        overrides['iou_threshold'] = args.iou
    if args.dish_shrink is not None:
        overrides['dish_shrink'] = args.dish_shrink
    if overrides:
        cfg = replace(cfg, **overrides)
    result = run_batch(args.dataset,args.output,cfg,max_images=args.limit,
                       save_images=not args.no_images,
                       save_evaluations=not args.no_images,save_masks=args.save_masks,
                       progress=lambda done,total,name,status:
                           print(f'[{done}/{total}] {status.upper():5} {name}',flush=True))
    print(f'Processed {result["processed"]}/{result["total"]} images; errors: {len(result["errors"])}')
    for metric in result['summary']:
        print(f'{metric["Method"]:8} GT={metric["GT"]} TP={metric["TP"]} '
              f'FP={metric["FP"]} FN={metric["FN"]} F1={metric["F1"]:.3f}')
    return 0 if not result['errors'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
