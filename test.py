import torch
from torch import nn

from nicegui import run, ui
from label_anything.models.build_lam import LabelAnything

    
print("Loading model...")
model = LabelAnything.from_pretrained("pasqualedem/label_anything_sam_1024_coco").model
print("Model loaded.")
image_enncoder = model.image_encoder

def compute_sum(tensor) -> float:
    with torch.no_grad():
        return image_enncoder(tensor).sum().item()

async def handle_click():
    result = await run.cpu_bound(compute_sum, torch.randn(1, 3, 1024, 1024))  # Example tensor
    print(f"Sum is {result}")
    ui.notify(f'Sum is {result}')

ui.button('Compute', on_click=handle_click)

ui.run()