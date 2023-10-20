import importlib
from typing import Mapping

from super_gradients import KDTrainer
from super_gradients.common import StrictLoad
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model

from logger.text_logger import get_logger

logger = get_logger(__name__)


class KDSegTrainer(SegmentationTrainer, KDTrainer):
    """
    Knowledge Distillation Trainer.
    """

    def init_model(self, params: Mapping, resume: bool, checkpoint_path: str = None):
        logger.info("Initializing teacher model")
        self.teacher_architecture, _ = self._load_model(
            {**params, "model": params["kd"]["teacher"]}
        )
        self.net = self.teacher_architecture
        self._net_to_device()
        self.teacher_architecture = self.net
        teacher_checkpoint = (
            params.get("kd", {}).get("teacher", {}).get("checkpoint_path", None)
        )
        if teacher_checkpoint is not None:
            load_checkpoint_to_model(
                params["kd"]["teacher"]["checkpoint_path"],
                load_backbone=False,
                net=self.teacher_architecture,
                strict=StrictLoad.ON.value,
                load_weights_only=False,
            )
        else:
            logger.warning("No teacher checkpoint provided, using random weights")
        logger.info("Initializing student model")
        super().init_model(params, resume, checkpoint_path)

        self.net = self._load_kd_module(
            params["kd"]["module"], student=self.net, teacher=self.teacher_architecture
        )

    def _save_best_checkpoint(self, epoch, state):
        """
        Overrides parent best_ckpt saving to modify the state dict so that we only save the student.
        """
        if self.ema:
            best_net = self.ema_model.ema.module.student
            state.pop("ema_net")
        else:
            best_net = self.net.module.student

        state["net"] = best_net.state_dict()
        self.sg_logger.add_checkpoint(
            tag=self.ckpt_best_name, state_dict=state, global_step=epoch
        )

    def _restore_best_params(self):
        self.checkpoint = load_checkpoint_to_model(
            ckpt_local_path=f"{self.checkpoints_dir_path}/ckpt_best.pth",
            load_backbone=False,
            net=self.net.module.student,
            strict=StrictLoad.ON.value,
            load_weights_only=True,
        )

    def _load_kd_module(self, model_params, student, teacher):
        arch_params = model_params["params"]
        try:
            module, model_cls = get_module_class_from_path(model_params["name"])
            model_module = importlib.import_module(module)
            model_cls = getattr(model_module, model_cls)
            model = model_cls(arch_params, student=student, teacher=teacher)
        except (AttributeError, ValueError):
            if model_params["name"] in KD_MODELS_DICT.keys():
                model = KD_MODELS_DICT[model_params["name"]](
                    arch_params, student=student, teacher=teacher
                )
            else:
                model = KD_ARCHITECTURES[model_params["name"]](
                    arch_params, student=student, teacher=teacher
                )
        return model
