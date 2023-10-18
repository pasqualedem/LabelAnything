from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.utils import HpmStruct


class LogitsDistillationModule(KDModule):
    """
    Logits Distillation Module
    """
    def __init__(self, arch_params, student, teacher, **kwargs):
        super().__init__(arch_params, student, teacher, **kwargs)
        if hasattr(self.student.module, "update_param_groups"):
            self.update_param_groups = self.student_update_param_groups
        else:
            self.update_param_groups = self.default_update_param_groups

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        if hasattr(self.student.module, "initialize_param_groups"):
            return self.student.module.initialize_param_groups(lr, training_params)
        else:
            return [{"named_params": self.student.named_parameters()}]

    def default_update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct,
                            total_batch: int) -> list:  
        for param_group in param_groups:
            param_group["lr"] = lr
        return param_groups

    def student_update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct,
                            total_batch: int) -> list:
        return self.student.update_param_groups(param_groups, lr, epoch, iter, training_params, total_batch)
    