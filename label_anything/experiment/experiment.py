from __future__ import annotations

import copy
import gc
import os
import glob
import uuid
import pandas as pd
from typing import Mapping
from easydict import EasyDict

from label_anything.experiment.run import Run, ParallelRun
from label_anything.experiment.resume import (
    ExpLog,
    get_interrupted_run,
    retrieve_run_to_resume,
)
from label_anything.utils.utils import get_timestamp, load_yaml, nested_dict_update, update_collection
from label_anything.utils.grid import linearize, linearized_to_string, make_grid
from label_anything.utils.optuna import Optunizer
from label_anything.logger.text_logger import get_logger

logger = get_logger(__name__)


class GridSummary:
    def __init__(
        self,
        total_runs,
        total_runs_excl_grid,
        total_runs_to_run,
        total_runs_excl,
    ):
        self.total_runs = total_runs
        self.total_runs_excl_grid = total_runs_excl_grid
        self.total_runs_to_run = total_runs_to_run
        self.total_runs_excl = total_runs_excl

    def update(self, d):
        self.total_runs = d.get("total_runs") or self.total_runs
        self.total_runs_excl_grid = (
            d.get("total_runs_excl_grid") or self.total_runs_excl_grid
        )
        self.total_runs_to_run = d.get("total_runs_to_run") or self.total_runs_to_run
        self.total_runs_excl = d.get("total_runs_excl") or self.total_runs_to_run


class ExpSettings(EasyDict):
    def __init__(self, *args, **kwargs):
        self.start_from_grid = 0
        self.start_from_run = 0
        self.resume = False
        self.resume_last = False
        self.tracking_dir = ""
        self.excluded_files = ""
        self.name = ""
        self.group = ""
        self.continue_with_errors = True
        self.logger = None
        self.search = "grid"
        self.direction = None
        self.n_trials = None
        self.max_parallel_runs = 1
        self.uuid = None
        self.timestamp = get_timestamp()
        super().__init__(*args, **kwargs)
        self.tracking_dir = self.tracking_dir or ""

    def update(self, e: ExpSettings, **f):
        if e is None:
            return
        self.start_from_grid = e.start_from_grid or self.start_from_grid
        self.start_from_run = e.start_from_run or self.start_from_run
        self.resume = e.resume or self.resume
        self.resume_last = e.resume_last or self.resume_last
        self.tracking_dir = e.tracking_dir or self.tracking_dir
        self.excluded_files = e.excluded_files or self.excluded_files
        self.group = e.group or self.group
        self.logger = e.logger or self.logger
        self.continue_with_errors = (
            not e.continue_with_errors or self.continue_with_errors
        )
        self.search = e.search or self.search
        self.direction = e.direction or self.direction
        self.n_trials = e.n_trials or self.n_trials
        self.max_parallel_runs = e.max_parallel_runs or self.max_parallel_runs
        self.uuid = e.uuid or self.uuid


class Status:
    STARTING = "starting"
    CRASHED = "crashed"
    FINISHED = "finished"

    def __init__(
        self, grid, run, params, n_grids, grid_len, run_name=None, run_url=None
    ):
        self.grid = grid
        self.run = run
        self.params = params
        self.status = self.STARTING
        self.grid_len = grid_len
        self.n_grids = n_grids
        self.exception = None
        self.run_name = run_name
        self.run_url = run_url

    def finish(self):
        self.params = {}
        self.status = self.FINISHED
        return self

    def crash(self, exception):
        self.status = self.CRASHED
        self.exception = exception
        return self


class StatusManager:
    def __init__(self, n_grids, max_parallel_runs=1):
        self.n_grids = n_grids
        self.cur_status = None
        self.max_parallel_runs = max_parallel_runs
        self.cur_parallel_runs = 0

    def new_run(self, grid, run, params, grid_len, run_name=None, run_url=None):
        self.cur_status = Status(
            grid=grid,
            run=run,
            params=params,
            n_grids=self.n_grids,
            grid_len=grid_len,
            run_name=run_name,
            run_url=run_url,
        )
        self.cur_parallel_runs += 1
        return self.cur_status

    def update_run(self, run_name, run_url):
        self.cur_status.run_name = run_name
        self.cur_status.run_url = run_url
        return self.cur_status

    def finish_run(self):
        self.cur_parallel_runs -= 1
        return self.cur_status.finish()

    def crash_run(self, exception):
        return self.cur_status.crash(exception)


class Experimenter:
    EXP_FINISH_SEP = "#" * 50 + " FINISHED " + "#" * 50 + "\n"
    EXP_CRASHED_SEP = "|\\" * 50 + "CRASHED" + "|\\" * 50 + "\n"

    def __init__(self):
        self.gs = None
        self.exp_settings = ExpSettings()
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings["parameters"]
        other_grids = settings["other_grids"]
        self.exp_settings = ExpSettings(settings["experiment"])
        if track_dir := self.exp_settings["tracking_dir"]:
            os.makedirs(track_dir, exist_ok=True)

        print("\n" + "=" * 100)
        complete_grids = [base_grid]
        if other_grids:
            complete_grids += [
                nested_dict_update(copy.deepcopy(base_grid), other_run)
                for other_run in other_grids
            ]
        logger.info(f"There are {len(complete_grids)} grids")

        if self.exp_settings.search == "grid":
            return self.generate_grid_search(complete_grids, other_grids)
        elif self.exp_settings.search == "optim":
            return self.generate_optim_search(complete_grids)
        else:
            raise ValueError(f"Unknown search type: {self.exp_settings.search}")

    def generate_optim_search(self, complete_grids):
        fname = f"{self.exp_settings.name}_{self.exp_settings.group.replace('/', '_')}"
        study_names = [f"{fname}_{i}" for i in range(len(complete_grids))]
        self.grids = [
            Optunizer(
                study_name=name,
                grid=grid,
                storage_base=self.exp_settings.tracking_dir,
                n_trials=self.exp_settings.n_trials,
                direction=self.exp_settings.direction,
            )
            for name, grid in zip(study_names, complete_grids)
        ]
        self.generate_grid_summary()

    def generate_grid_search(self, complete_grids, other_grids):
        self.grids, dot_elements = zip(
            *[
                make_grid(grid, return_cartesian_elements=True)
                for grid in complete_grids
            ]
        )
        # WARNING: Grids' objects have the same IDs!
        dot_elements = list(dot_elements)
        if len(dot_elements) > 1:
            dot_elements[1:] = [
                list(dict(linearize(others) + dot).items())
                for others, dot in zip(other_grids, dot_elements[1:])
            ]

        # Modify starting grid and run to manage the resume
        self.manage_resume()

        for i, grid in enumerate(self.grids):
            info = f"Found {len(grid)} runs from grid {i}"
            last_grid = (
                self.exp_settings.start_from_grid
                if self.exp_settings.start_from_grid is not None
                else len(self.grids)
            )
            if i < last_grid:
                info += f", skipping grid {i} with {len(grid)} runs"
            logger.info(info)
        self.generate_grid_summary()

        if self.exp_settings.excluded_files:
            os.environ["WANDB_IGNORE_GLOBS"] = self.exp_settings.excluded_files

        print_preview(self, self.gs, self.grids, dot_elements)
        print("=" * 100 + "\n")

        return self.gs, self.grids, dot_elements

    def generate_grid_summary(self):
        total_runs = sum(len(grid) for grid in self.grids)
        if self.exp_settings.start_from_grid is None:
            total_runs_excl_grid = total_runs - len(self.grids[-1])
            total_runs_excl = total_runs
        else:
            total_runs_excl_grid = total_runs - sum(
                len(grid) for grid in self.grids[self.exp_settings.start_from_grid :]
            )
            total_runs_excl = total_runs_excl_grid + self.exp_settings.start_from_run
        total_runs_to_run = total_runs - total_runs_excl
        self.gs = GridSummary(
            total_runs=total_runs,
            total_runs_excl_grid=total_runs_excl_grid,
            total_runs_to_run=total_runs_to_run,
            total_runs_excl=total_runs_excl,
        )

    def execute_runs_generator(self):
        exp_log = ExpLog(
            self.exp_settings["tracking_dir"],
            self.exp_settings.name,
            self.exp_settings.group,
        )
        starting_run = self.exp_settings.start_from_run
        status_manager = StatusManager(len(self.grids))
        if self.exp_settings.resume_last and self.exp_settings.search == "grid":
            logger.info("+ another run to finish!")
            grid_list = [
                (i, j)
                for i in range(len(self.grids))
                for j in range(len(self.grids[i]))
            ]
            if self.exp_settings.start_from_grid is None:
                grid_len = len(self.grids[-1])
                sg, sr = grid_list[-1]
            else:
                grid_len = len(self.grids[self.exp_settings.start_from_grid])
                index = grid_list.index(
                    (
                        self.exp_settings.start_from_grid,
                        self.exp_settings.start_from_run,
                    )
                )
                sg, sr = grid_list[index - 1]
            try:
                exp_log.insert_run(sg, sr)
                run = get_interrupted_run(self.exp_settings)
                yield status_manager.new_run(
                    sg,
                    sr,
                    run.params,
                    grid_len,
                    run.seg_trainer.sg_logger.name,
                    run.seg_trainer.sg_logger.url,
                )
                logger.info(f"Running grid {sg} out of {len(self.grids) - 1}")
                logger.info(
                    f"Running run {sr - 1} out of {grid_len} ({sum(len(self.grids[k]) for k in range(sg)) + sr} / {self.gs.total_runs - 1})"
                )
                run.launch()
                print(self.EXP_FINISH_SEP)
                exp_log.finish_run(sg, sr)
                yield status_manager.finish_run()
            except Exception as ex:
                logger.error(f"Experiment {sg} failed with error {ex}")
                print(self.EXP_CRASHED_SEP)
                exp_log.finish_run(sg, sr, crashed=True)
                if not self.exp_settings.continue_with_errors:
                    raise ex
                yield status_manager.crash_run(ex)
        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    exp_log.insert_run(i, j)
                    yield status_manager.new_run(i, j, params, len(grid))
                    logger.info(f"Running grid {i} out of {len(self.grids) - 1}")
                    logger.info(
                        f"Running run {j} out of {len(grid) - 1} ({sum(len(self.grids[k]) for k in range(i)) + j} / {self.gs.total_runs - 1})"
                    )
                    run = Run()
                    run.init({"experiment": {**self.exp_settings}, **params})
                    yield status_manager.update_run(
                        run.name,
                        run.url,
                    )
                    metric = run.launch()
                    print(self.EXP_FINISH_SEP)
                    exp_log.finish_run(i, j)
                    if self.exp_settings.search == "optim":
                        self.grids[i].report_result(metric)
                    gc.collect()
                    yield status_manager.finish_run()
                except Exception as ex:
                    logger.error(f"Experiment {i} failed with error {ex}")
                    print(self.EXP_CRASHED_SEP)
                    exp_log.finish_run(i, j, crashed=True)
                    if not self.exp_settings.continue_with_errors:
                        raise ex
                    yield status_manager.crash_run(ex)

    def execute_runs(self, only_create=False):
        for _ in self.execute_runs_generator():
            pass

    def manage_resume(self):
        if self.exp_settings.resume:
            (
                self.exp_settings.start_from_grid,
                self.exp_settings.start_from_run,
                self.exp_settings.resume_last,
            ) = retrieve_run_to_resume(self.exp_settings, self.grids)
        else:
            self.exp_settings.resume_last = False

    def update_settings(self, d):
        self.exp_settings = update_collection(self.exp_settings, d)
        if self.gs is None:
            return
        self.gs.update(self.exp_settings)
        if "resume" in d:
            self.manage_resume()
            self.generate_grid_summary()


class ParallelExperimenter(Experimenter):
    EXP_FINISH_SEP = "#" * 50 + " LAUNCHED " + "#" * 50 + "\n"
    EXP_CRASHED_SEP = "|\\" * 50 + "CRASHED" + "|\\" * 50 + "\n"

    def __init__(self):
        super().__init__()

    def execute_runs_generator(self, only_create=False):
        exp_log = ExpLog(
            self.exp_settings["tracking_dir"],
            self.exp_settings.name,
            self.exp_settings.group,
        )
        starting_run = self.exp_settings.start_from_run
        self.exp_settings.uuid = self.exp_settings.uuid or str(uuid.uuid4())[:8]
        status_manager = StatusManager(
            len(self.grids), self.exp_settings.max_parallel_runs
        )
        if self.exp_settings.resume_last and self.exp_settings.search == "grid":
            logger.info("+ another run to finish!")
            grid_list = [
                (i, j)
                for i in range(len(self.grids))
                for j in range(len(self.grids[i]))
            ]
            if self.exp_settings.start_from_grid is None:
                grid_len = len(self.grids[-1])
                sg, sr = grid_list[-1]
            else:
                grid_len = len(self.grids[self.exp_settings.start_from_grid])
                index = grid_list.index(
                    (
                        self.exp_settings.start_from_grid,
                        self.exp_settings.start_from_run,
                    )
                )
                sg, sr = grid_list[index - 1]
            try:
                exp_log.insert_run(sg, sr)
                run = get_interrupted_run(self.exp_settings)
                yield status_manager.new_run(
                    sg,
                    sr,
                    run.params,
                    grid_len,
                    None,
                    None,
                )
                logger.info(f"Running grid {sg} out of {len(self.grids) - 1}")
                logger.info(
                    f"Running run {sr - 1} out of {grid_len} ({sum(len(self.grids[k]) for k in range(sg)) + sr} / {self.gs.total_runs - 1})"
                )
                run.launch(only_create=only_create)
                print(self.EXP_FINISH_SEP)
                exp_log.finish_run(sg, sr)
                yield status_manager.finish_run()
            except Exception as ex:
                logger.error(f"Experiment {sg} failed with error {ex}")
                print(self.EXP_CRASHED_SEP)
                exp_log.finish_run(sg, sr, crashed=True)
                if not self.exp_settings.continue_with_errors:
                    raise ex
                yield status_manager.crash_run(ex)
        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    exp_log.insert_run(i, j)
                    yield status_manager.new_run(i, j, params, len(grid))
                    logger.info(f"Running grid {i} out of {len(self.grids) - 1}")
                    logger.info(
                        f"Running run {j} out of {len(grid) - 1} ({sum(len(self.grids[k]) for k in range(i)) + j} / {self.gs.total_runs - 1})"
                    )
                    run = ParallelRun(
                        experiment_timestamp=self.exp_settings.timestamp,
                        params={"experiment": {**self.exp_settings}, **params},
                    )
                    metric = run.launch(only_create=only_create)
                    print(self.EXP_FINISH_SEP)
                    exp_log.finish_run(i, j)
                    if self.exp_settings.search == "optim":
                        self.grids[i].report_result(metric)
                    gc.collect()
                    yield status_manager.finish_run()
                except Exception as ex:
                    logger.error(f"Experiment {i} failed with error {ex}")
                    print(self.EXP_CRASHED_SEP)
                    exp_log.finish_run(i, j, crashed=True)
                    if not self.exp_settings.continue_with_errors:
                        raise ex
                    yield status_manager.crash_run(ex)

    def execute_runs(self, only_create=False):
        for _ in self.execute_runs_generator(only_create=only_create):
            pass


def experiment(
    param_path: str = "parameters.yaml",
    parallel: bool = False,
    only_create: bool = False,
    preview: bool = False,
):
    logger.info("Running experiment")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")

    experimenter = ParallelExperimenter() if parallel or only_create else Experimenter()
    experimenter.calculate_runs(settings)
    if not preview:
        experimenter.execute_runs(only_create=only_create)


def run(param_path: str = "parameters.yaml"):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    single_run = Run()
    single_run.init(settings)
    single_run.launch()
    
    
def validate(param_path: str = "parameters.yaml", generate_json: bool = False):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    single_run = Run()
    single_run.init(settings)
    epoch = 0
    with single_run.plat_logger.validate():
        single_run.validate(epoch=epoch, generate_json=generate_json)
    
    
def test(param_path: str = "parameters.yaml"):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    single_run = Run()
    single_run.init(settings)
    single_run.test()
    single_run.end()


def preview(settings: Mapping, param_path: str = "local variable"):
    print(f"Loaded parameters from {param_path}")

    experimenter = Experimenter()
    _, _, _ = experimenter.calculate_runs(settings)


def print_preview(experimenter, grid_summary, grids, cartesian_elements):
    summary_series = pd.concat(
        [pd.Series(grid_summary), pd.Series(experimenter.exp_settings.__dict__)]
    )
    summary_string = f"\n{summary_series.to_string()}\n"

    dfs = [
        pd.DataFrame(
            linearized_to_string(dot_element),
            columns=[f"Grid {i}", f"N. runs: {len(grid)}"],
        )
        for i, (dot_element, grid) in enumerate(zip(cartesian_elements, grids))
    ]
    mark_grids = "\n\n".join(df.to_string(index=False) for df in dfs)
    mark_grids = "Most important parameters for each grid \n" + mark_grids
    logger.info(f"\n{summary_string}\n{mark_grids}")
