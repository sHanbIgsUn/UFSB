import os
from config import cfg_base as cfg
import argparse
from datasets.make_dataloader import make_dataloader
from datasets.make_dataloader_clipreid import make_dataloader_pf
from model.make_model import make_model
from model.make_model_clipreid_pf import make_model_pf
from module.domain_discriminator import DomainDiscriminator
from processor.processor import do_inference
from processor.processor_clipreid_stage2_mt import do_inference_pf
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid_pf.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader_stage2_person, train_loader_stage2_face, train_loader_stage1_person, train_loader_stage1_face, val_loader_person, val_loader_face, num_query_person, num_query_face, num_classes_person, num_classes_face, cam_num_person, cam_num_face, view_num_person,view_num_face,real_id_person,real_id_face = make_dataloader_pf(cfg)
    model = make_model_pf(cfg, [num_classes_person, num_classes_face], [cam_num_person, cam_num_face], [view_num_person,view_num_face],['person','face'],[real_id_person,real_id_face]).to('cpu')

    domainDiscriminator_person=DomainDiscriminator(in_feature=model.in_planes, hidden_size=1024,proj_feature=model.in_planes_proj,sigmoid=False)
    domainDiscriminator_face=DomainDiscriminator(in_feature=model.in_planes, hidden_size=1024,proj_feature=model.in_planes_proj,sigmoid=False)

    model.load_param(cfg.TEST.WEIGHT)



    do_inference_pf(cfg,
                model,
                [val_loader_person,val_loader_face],
                [num_query_person,num_query_face])
    



