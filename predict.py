import tempfile
from pathlib import Path

import cog
import torch
from torchvision.utils import save_image

from transfer import WCT2
from utils.io import Timer, load_segment, open_image

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class Predictor(cog.Predictor):
    def setup(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @cog.input("content", type=Path, help="content image")
    @cog.input("style", type=Path, help="style image")
    @cog.input(
        "content_segment",
        type=Path,
        default=None,
        help="content segment label image (optional)",
    )
    @cog.input(
        "style_segment",
        type=Path,
        default=None,
        help="style segment label image (optional)",
    )
    @cog.input(
        "option_unpool",
        type=str,
        default="cat5",
        options=["sum", "cat5"],
        help="model version",
    )
    @cog.input(
        "transfer_at_encoder",
        type=bool,
        default=True,
        help="stylize at the encoder module",
    )
    @cog.input(
        "transfer_at_decoder",
        type=bool,
        default=True,
        help="stylize at the decoder module",
    )
    @cog.input(
        "transfer_at_skip",
        type=bool,
        default=True,
        help="stylize at the skipped high frequency components",
    )
    @cog.input("image_size", type=int, default=512, help="output image size")
    @cog.input(
        "alpha",
        type=float,
        default=1,
        help="blending ratio between content and stylized features",
    )
    def predict(
        self,
        content,
        style,
        content_segment=None,
        style_segment=None,
        option_unpool="cat5",
        transfer_at_encoder=True,
        transfer_at_decoder=True,
        transfer_at_skip=True,
        image_size=512,
        alpha=1,
    ):

        assert (
            transfer_at_skip or transfer_at_encoder or transfer_at_decoder
        ), "at least one of transfer_at_encoder, transfer_at_decoder, transfer_at_skip needs to be True"

        fnames = [str(content), str(style)]
        if content_segment is not None:
            fnames.append(str(content_segment))
        if style_segment is not None:
            fnames.append(str(style_segment))
        for fname in fnames:
            assert is_image_file(fname), f" {fname} is not valid image"

        transfer_at = set()

        if transfer_at_encoder:
            transfer_at.add("encoder")
        if transfer_at_decoder:
            transfer_at.add("decoder")
        if transfer_at_skip:
            transfer_at.add("skip")

        content = open_image(str(content), image_size).to(self.device)
        style = open_image(str(style), image_size).to(self.device)
        content_segment = (
            load_segment(str(content_segment), image_size)
            if content_segment is not None
            else None
        )
        style_segment = (
            load_segment(str(style_segment), image_size)
            if style_segment is not None
            else None
        )

        with Timer("Elapsed time in whole WCT: {}", True):
            wct2 = WCT2(
                transfer_at=transfer_at, option_unpool=option_unpool, device=self.device
            )
            with torch.no_grad():
                img = wct2.transfer(
                    content, style, content_segment, style_segment, alpha=alpha
                )
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            save_image(img.clamp_(0, 1), str(out_path), padding=0)

        return out_path
