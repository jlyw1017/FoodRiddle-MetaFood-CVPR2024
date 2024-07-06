"""Main entrance for food reconstruction."""
import argparse
import logging
import os.path

from data_utils.data_loader import MTFDataSet, ObjType
from core import colmap_runner
from core.data_preprocess import preprocess_input_data
from core.food_scene_creater import create_obj, enlarge_roi


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_parser():
    """Parser definition."""
    desc = "Command line argument paser."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("-d", '--dataset_root', required=True,
                        help="Input folder of dataset,"
                        " e.g. ../MTF_Challenge")
    parser.add_argument("-i", '--index', required=False,
                        help="the obj index, if set run only the given index.")
    parser.add_argument("-sk", '--skip_preprocess', action='store_true',
                        help="If true skip the preprocess.")
    parser.add_argument("-t", '--type', required=False,
                        type=ObjType, choices=list(ObjType),
                        help="use simple, medium or hard to filter task.")
    return parser


def main():
    """The main process of the reconstruction."""
    parser = get_parser()
    args = parser.parse_args()

    data_set = MTFDataSet(args.dataset_root)

    if args.index is not None:
        metas = [data_set.get_meta_by_index(args.index)]
    elif args.type is not None:
        metas = data_set.get_meta_by_type(args.type)
    else:
        metas = data_set.get_all_metas()

    logger.info(f"Start reconstruction for {len(metas)} objs.")

    scene_by_index = {}
    for meta in metas:
        logger.info(f"Process obj {meta.obj_index} {meta.obj_name}")

        if not args.skip_preprocess:
            logger.info("Preprocess depth and mask.")
            preprocess_input_data(meta)

        scene_by_index[meta.obj_index] = create_obj(meta)

    for food_scene in scene_by_index.values():
        if not args.skip_preprocess:
            logger.info("Enlarge ROI.")
            # Skip hard task bcz there is only one frame.
            if food_scene.meta.obj_type == ObjType.HARD:
                enlarge_roi(food_scene)

    # Run Colmap reconstruction.
    for meta in metas:
        # Skip hard task bcz there is only one frame.
        if meta.obj_type == ObjType.HARD:
            continue

        colmapper = colmap_runner.ColmapProcessor()
        colmapper.run_colmap(meta.image_folder_path,
                             os.path.join(meta.obj_root, "colmap_result"))


if __name__ == '__main__':
    main()
