import os
import hydra
import pytorch_lightning as pl
from detectron2.utils.events import EventStorage # type: ignore
from habitat_learn_od.utils.train_helpers import get_training_params
from habitat_learn_od.utils import data_modules, teacher_student_modules
from common.env_utils.object_detector_sensors import * 


@hydra.main(config_path="../config", config_name="train.yaml")
def main(cfg):
    data_path = os.path.join(os.getcwd(), "data")
    if not (os.path.exists(data_path)):
        os.symlink(cfg.data_base_dir, data_path)

    config_path = os.path.join(os.getcwd(), "config")
    if not (os.path.exists(config_path)):
        os.symlink(cfg.cfg_base_dir, config_path)

    tp_path = os.path.join(os.getcwd(), "third_party")
    if not (os.path.exists(tp_path)):
        os.symlink(cfg.tp_base_dir, tp_path)

    teacher_student = teacher_student_modules.TeacherStudent(**cfg,**cfg.training, **cfg.detic_args, device="cuda:0")
    trainer_config = get_training_params(cfg)
    print(trainer_config)
    dataset_path = os.path.join(os.getcwd(), "data", cfg.collected_set)
    trainer = pl.Trainer(**trainer_config)    

    checkpoint_path = None


    with EventStorage(start_iter=0) as storage:
        for id_iteration in range(cfg.training.n_iterations):
            if 'use_gt' in cfg.training and cfg.training['use_gt']:
                dm = data_modules.GTDataModule(
                    pseudo_labeler=teacher_student.pseudo_labeler,
                    collection_policy=None,
                    dataset_path=dataset_path,
                    **cfg, # type: ignore
                    **cfg.training
                )
            else:
                dm = data_modules.HabitatDataModule(
                    pseudo_labeler=teacher_student.pseudo_labeler,
                    collection_policy=None,
                    dataset_path=dataset_path,
                    **cfg, # type: ignore
                    **cfg.training
                )

            if checkpoint_path is not None:
                teacher_student.load_from_checkpoint(checkpoint_path)
            
            # if id_iteration == 0:
            #     trainer.validate(model=teacher_student, datamodule=dm)

            trainer.fit(model=teacher_student, datamodule=dm)

            checkpoint_path = f"iteration-{id_iteration}.ckpt"
            trainer.save_checkpoint(checkpoint_path)

            trainer_config['max_epochs'] += cfg.training.epochs_per_iteration
            if "update_target" in cfg.training and cfg.training['update_target']:
                if not "ema" in cfg.training or not cfg.training.ema:
                    teacher_student.pseudo_labeler.reinit(teacher_student.online_network)


if __name__ == "__main__":
    main()