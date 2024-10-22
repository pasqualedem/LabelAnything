import copy
import click

from label_anything.utils.grid import linearize, make_grid
from label_anything.utils.utils import get_timestamp, load_yaml, nested_dict_update
from lora.sample import main as lora_sample_main
from lora.validate import main as lora_validate_main


from label_anything.experiment.run import ParallelRun


class ParallelLoraRun(ParallelRun):

    slurm_script = "slurm/launch_lora"


def parallel_experiment_lora(param_file):
    timestamp = get_timestamp()
    settings = load_yaml(param_file)
    base_grid = settings["parameters"]
    other_grids = settings["other_grids"]

    print("\n" + "=" * 100)
    complete_grids = [base_grid]
    if other_grids:
        complete_grids += [
            nested_dict_update(copy.deepcopy(base_grid), other_run)
            for other_run in other_grids
        ]

        grids, dot_elements = zip(
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

        for i, grid in enumerate(grids):
            print(f"Grid {i+1}:")
            for j, run_params in enumerate(grid):
                print(f"Run {j+1}:")
                run_params["experiment"] = {"group": "LoRa"}
                run = ParallelLoraRun(run_params, experiment_timestamp=timestamp)
                run.launch()


@click.command()
@click.option("--num_iterations", default=10, help="Number of iterations to run.")
@click.option("--device", default="cuda", help="Device to use (e.g., cuda or cpu).")
@click.option("--lora_r", default=32, help="LoRA r parameter.", type=int)
@click.option("--lora_alpha", default=32.0, help="LoRA alpha parameter.", type=float)
@click.option("--lr", default=1e-4, help="Learning rate.", type=float)
@click.option("--model", default="label_anything", help="Model to use.")
@click.option(
    "--target_modules",
    default="query,value",
    help="Comma-separated list of target modules.",
)
@click.option(
    "--substitutor",
    default="default",
    help="Substitutor to use. (default or incremental)",
)
@click.option(
    "--n_ways", default=2, help="Number of classes of the FSS validation", type=int
)
@click.option(
    "--k_shots",
    default=5,
    help="Number of shots per class of the FSS validation",
    type=int,
)
@click.option("--val_num_samples", default=100, help="Number of samples for validation.")
@click.option("--lora_dropout", default=0.1, help="LoRA dropout value.", type=float)
@click.option(
    "--experiment_file",
    default=None,
    help="Path to the file containing the parameters for the experiment, launching multiple parallel runs",
)
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
def cli(
    num_iterations,
    device,
    lora_r,
    lora_alpha,
    lr,
    model,
    target_modules,
    lora_dropout,
    substitutor,
    n_ways,
    k_shots,
    val_num_samples,
    experiment_file,
    parameters,
):
    """
    Command-line interface for setting parameters for LoRA training or testing.
    Collects parameters and passes them as a dictionary.
    """
    if experiment_file is not None:
        parallel_experiment_lora(experiment_file)
        return

    if parameters is not None:
        params = load_yaml(parameters)
        params["target_modules"] = params["target_modules"].split(",")
    else:
        # Convert the target_modules string to a list
        target_modules_list = target_modules.split(",")

        # Create the parameters dictionary
        params = {
            "num_iterations": num_iterations,
            "device": device,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lr": lr,
            "target_modules": target_modules_list,
            "lora_dropout": lora_dropout,
            "substitutor": substitutor,
            "n_ways": n_ways,
            "k_shots": k_shots,
            "val_num_samples": val_num_samples,
            "model": model,
        }

    # Call the main function with the dictionary of parameters
    main(params)


def main(params):
    """
    Main function that accepts a dictionary of parameters.
    """
    # Print the parameters for debugging
    click.echo("Running with the following parameters:")
    for key, value in params.items():
        click.echo(f"{key}: {value}")

    lora_validate_main(params)


if __name__ == "__main__":
    cli()
