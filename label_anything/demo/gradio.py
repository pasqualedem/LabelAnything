import gradio as gr


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.ImageMask(label="Input Image", brush=gr.Brush(size=20, color_mode="rgb", color="black"))

def main():
    demo.launch(inbrowser=True)
