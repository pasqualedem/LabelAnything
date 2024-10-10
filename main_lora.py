from lora.sample import main as lora_sample_main
from lora.validate import main as lora_validate_main


import click

@click.command()
@click.option('--num_iterations', default=10, help='Number of iterations to run.')
@click.option('--device', default='cuda', help='Device to use (e.g., cuda or cpu).')
@click.option('--lora_r', default=32, help='LoRA r parameter.', type=int)
@click.option('--lora_alpha', default=32.0, help='LoRA alpha parameter.', type=float)
@click.option('--lr', default=1e-4, help='Learning rate.', type=float)
@click.option('--target_modules', default='query,value', help='Comma-separated list of target modules.')
@click.option('--lora_dropout', default=0.1, help='LoRA dropout value.', type=float)
def cli(num_iterations, device, lora_r, lora_alpha, lr, target_modules, lora_dropout):
    """
    Command-line interface for setting parameters for LoRA training or testing.
    Collects parameters and passes them as a dictionary.
    """
    # Convert the target_modules string to a list
    target_modules_list = target_modules.split(',')
    
    # Create the parameters dictionary
    params = {
        'num_iterations': num_iterations,
        'device': device,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lr': lr,
        'target_modules': target_modules_list,
        'lora_dropout': lora_dropout
    }

    # Call the main function with the dictionary of parameters
    main(params)

def main(params):
    """
    Main function that accepts a dictionary of parameters.
    """
    # Print the parameters for debugging
    click.echo('Running with the following parameters:')
    for key, value in params.items():
        click.echo(f'{key}: {value}')

    lora_validate_main(params)
    
if __name__ == '__main__':
    cli()
