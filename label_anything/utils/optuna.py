import optuna

from label_anything.utils.grid import linearize, delinearize


class Optunizer:
    def __init__(
        self, study_name, grid, n_trials, storage_base="optuna", direction="maximize"
    ):
        direction = (
            "maximize"
            if direction == "max"
            else "minimize"
            if direction == "min"
            else direction
        )
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        storage_name = f"sqlite://{storage_base}/{study_name}.db"
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )

        self._init_grid(grid)
        self.trial = None

    def _init_grid(self, grid):
        self.lin_grid = linearize(grid)
        self.hyperparameters = [
            (key, value) for key, value in self.lin_grid if len(value) > 1
        ]
        self.parameters = [
            (key, value[0]) for key, value in self.lin_grid if len(value) == 1
        ]

    def next_trial(self):
        self.trial = self.study.ask()
        parameters = dict(self.parameters)
        for key, value in self.hyperparameters:
            name = key if isinstance(key, str) else ".".join(flat_tuple(key))
            parameters[key] = self._suggest(name, value)

        return delinearize(parameters)

    def __getitem__(self, item):
        return self.next_trial()

    def __len__(self):
        return self.n_trials

    def report_result(self, value):
        self.study.tell(self.trial, value)

    def _suggest(self, key, values):
        first_value = values[0]
        if isinstance(first_value, int):
            return self.trial.suggest_int(key, *values)
        elif isinstance(first_value, float):
            return self.trial.suggest_float(key, *values)
        else:
            return self.trial.suggest_categorical(key, choices=values)


def flat_tuple(t):
    if isinstance(t, tuple):
        if len(t) == 0:
            return ()
        return (t[0],) + flat_tuple(t[1])
    else:
        return (t,)
