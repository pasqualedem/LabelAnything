from collections import namedtuple
from inspect import signature

import torch
from super_gradients.training.models import SgModule
from super_gradients.training.utils import HpmStruct
from torchdistill.core.forward_hook import ForwardHookManager

from torch import nn

# from ezdl.models.layers.common import ConvModule
from utils.utils import unwrap_model_from_parallel
from models.kd.logits import LogitsDistillationModule


FDOutput = namedtuple(
    "FDOutput",
    [
        "student_features",
        "student_output",
        "teacher_features",
        "teacher_output",
    ],
)


class FeatureDistillationModule(LogitsDistillationModule):
    """
    Feature Distillation Module
    """

    def __init__(
        self,
        arch_params: HpmStruct,
        student: SgModule,
        teacher: torch.nn.Module,
        run_teacher_on_eval=False,
    ):
        """
        :param arch_params: architecture parameters
        :param student: student model
        :param teacher: teacher model
        :param run_teacher_on_eval: whether to run the teacher on eval
        """
        super().__init__(
            arch_params=arch_params,
            student=student,
            teacher=teacher,
            run_teacher_on_eval=run_teacher_on_eval,
        )

        if "hooks" in arch_params:
            self.teacher_hook_manager = ForwardHookManager(
                next(self.teacher.parameters()).device
            )
            self.student_hook_manager = ForwardHookManager(
                next(self.student.parameters()).device
            )
            self.add_hooks(arch_params["hooks"])
            self.hooks = True
        else:
            self.teacher_hook_manager = None
            self.student_hook_manager = None
            student, swrapped = unwrap_model_from_parallel(
                student, return_was_wrapped=True
            )
            teacher, twrapped = unwrap_model_from_parallel(
                teacher, return_was_wrapped=True
            )
            assert (
                "return_encoding" in signature(student.forward).parameters
            ), "Student model must return encoding if no hooks are used"
            assert (
                "return_encoding" in signature(teacher.forward).parameters
            ), "Teacher model must return encoding if no hooks are used"
            self.hooks = False
            if swrapped:
                self.student.forward = self.student.module.forward
            if twrapped:
                self.teacher.forward = self.teacher.module.forward

    def add_hooks(self, hooks):
        """
        Add hooks to the teacher and student
        :param hooks: list of hooks
        """
        if isinstance(hooks, dict):
            student_hooks = hooks["student"]
            teacher_hooks = hooks["teacher"]
        else:
            student_hooks = hooks
            teacher_hooks = hooks
        self._add_hooks_to_model(self.student, self.student_hook_manager, student_hooks)
        self._add_hooks_to_model(self.teacher, self.teacher_hook_manager, teacher_hooks)

    def _add_hooks_to_model(self, model, hook_manager, hooks):
        """
        Add hooks to a model
        :param model: model to add hooks to
        """
        model = unwrap_model_from_parallel(model)
        for hook in hooks:
            if isinstance(hook, str):
                hook_manager.add_hook(model, hook, requires_output=True)
            elif isinstance(hook, dict):
                hook_manager.add_hook(model, hook["name"], **hook["params"])
            else:
                raise ValueError("Invalid hook")

    def forward(self, x):
        """
        Forward pass
        :param x: input
        :return: output
        """
        return self.hook_forward(x) if self.hooks else self.stepped_forward(x)

    def hook_forward(self, x):
        student_output = self.student.forward(x)
        if self.teacher_input_adapter is not None:
            teacher_output = self.teacher.forward(self.teacher_input_adapter(x))
        else:
            teacher_output = self.teacher.forward(x)
        student_features = self.student_hook_manager.pop_io_dict()
        student_features = [
            student_features[hook]["output"] for hook in student_features
        ]
        teacher_features = self.teacher_hook_manager.pop_io_dict()
        teacher_features = [
            teacher_features[hook]["output"] for hook in teacher_features
        ]

        return FDOutput(
            student_features=student_features,
            student_output=student_output,
            teacher_features=teacher_features,
            teacher_output=teacher_output,
        )

    def stepped_forward(self, x):
        """
        Forward pass for a single step
        :param x: input
        :return: output
        """
        student_output, student_features = self.student.forward(x, return_encoding=True)
        if self.teacher_input_adapter is not None:
            teacher_output, teacher_features = self.teacher.forward(
                self.teacher_input_adapter(x), return_encoding=True
            )
        else:
            teacher_output, teacher_features = self.teacher.forward(
                x, return_encoding=True
            )
        return FDOutput(
            student_features=student_features,
            student_output=student_output,
            teacher_features=teacher_features,
            teacher_output=teacher_output,
        )


class FeatureDistillationConvAdapter(FeatureDistillationModule):
    def __init__(
        self,
        arch_params: HpmStruct,
        student: SgModule,
        teacher: torch.nn.Module,
        run_teacher_on_eval=False,
        epsilon=1e-6,
    ):
        super().__init__(
            arch_params=arch_params,
            student=student,
            teacher=teacher,
            run_teacher_on_eval=run_teacher_on_eval,
        )
        self.adapters = nn.ModuleList(
            [
                ConvModule(inp, out, 3, p="same")
                for inp, out in zip(
                    self.student.module.encoder_maps_sizes,
                    self.teacher.module.encoder_maps_sizes,
                )
            ]
        )

    def forward(self, x):
        fd_output = super().forward(x)
        student_features = [
            adapter(feat_map)
            for feat_map, adapter in zip(fd_output.student_features, self.adapters)
        ]
        return FDOutput(
            student_features=student_features,
            student_output=fd_output.student_output,
            teacher_features=fd_output.teacher_features,
            teacher_output=fd_output.teacher_output,
        )


class VariationalInformationDistillation(FeatureDistillationModule):
    """
    Feature Distillation Module that uses Variational Information
    """

    def __init__(
        self,
        arch_params: HpmStruct,
        student: SgModule,
        teacher: torch.nn.Module,
        run_teacher_on_eval=False,
        epsilon=1e-6,
    ):
        super().__init__(
            arch_params=arch_params,
            student=student,
            teacher=teacher,
            run_teacher_on_eval=run_teacher_on_eval,
        )
        self.mu_networks = nn.ModuleList(
            [
                ConvModule(inp, out, 3, p="same")
                for inp, out in zip(
                    self.student.module.encoder_maps_sizes,
                    self.teacher.module.encoder_maps_sizes,
                )
            ]
        )
        self.alphas = nn.ParameterList(
            [torch.nn.Parameter(torch.rand(1)) for _ in self.mu_networks]
        )
        self.sigmas = nn.ModuleList([SigmaVariance(epsilon) for _ in self.mu_networks])

    def forward(self, x):
        fd_output = super().forward(x)
        mu = [
            mu_network(feat_map)
            for feat_map, mu_network in zip(
                fd_output.student_features, self.mu_networks
            )
        ]
        sigmas = [sigma() for sigma in self.sigmas]
        return FDOutput(
            student_features=list(zip(mu, sigmas)),
            student_output=fd_output.student_output,
            teacher_features=fd_output.teacher_features,
            teacher_output=fd_output.teacher_output,
        )


class SigmaVariance(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self):
        return torch.nn.functional.softplus(self.sigma) + self.epsilon
