import lightning as L
from cyclopts import App
from lightning.pytorch import seed_everything

from src.configs import ModelVideoConfig
from src.data_module import AmongUsDatamodule
from src.models.fcos_pretrained import ModelFcosPretrained

app = App(name="Define Config for inferencing:")


@app.command
def run_inference_video(cfg: ModelVideoConfig = ModelVideoConfig()):
    """
    Run inference on all images in a directory and save results with bounding boxes

    Args:
        image_dir: directory containing images to process
        output_dir: directory to save output images with bounding boxes
        checkpoint_path: path to the model checkpoint file
        confidence_threshold: confidence threshold for drawing boxes
    """
    inference_cfg = cfg.video_cfg
    seed_everything(inference_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(
        cfg.datamodule_cfg, cfg.creation_cfg, cfg.transform_cfg, video_cfg=inference_cfg
    )
    trainer = L.Trainer(
        accelerator=inference_cfg.accelerator,
        enable_progress_bar=True,
    )
    model = ModelFcosPretrained.load_from_checkpoint(
        inference_cfg.checkpoint, weights_only=False
    )
    output = trainer.predict(model=model, datamodule=data_module)


if __name__ == "__main__":
    app()
