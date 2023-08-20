import os
import pathlib
import uuid
import base64
import imghdr
import requests

import gradio as gr

from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from PIL import Image
from io import BytesIO

from fastapi import Request, FastAPI, HTTPException

import modules.script_callbacks as script_callbacks
from modules.paths import Paths
import modules.shared as shared

from scripts.roop_logging import logger
from scripts.swapper import UpscaleOptions, swap_face, ImageResult, UpscalerData, FaceRestoration


class ImageSource(BaseModel):
    image_url: Optional[str]
    encoded_image: Optional[str]
    image_filepath: Optional[str]


class RequestUpscaleOptions(BaseModel):
    scale: int = 1
    upscaler_name: str = ""
    upscale_visibility: float = 1.0
    face_restorer_name: str = "CodeFormer"
    restorer_visibility: float = 1.0


class RoopOptions(BaseModel):
    task_id: str
    source_image: ImageSource
    swap_image: ImageSource
    face_index: int = 0
    model: str = "roop/inswapper_128.onnx"
    upscale_options: RequestUpscaleOptions = RequestUpscaleOptions()


class RoopResponse(BaseModel):
    task_id: str
    encoded_image: str
    start_time: str
    finish_time: str


def download_image(url: str, output_path: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Open the output file in binary mode, write the response content into it
    with open(output_path, 'wb') as out_file:
        out_file.write(response.content)

    # Determine the image type
    image_type = imghdr.what(output_path)

    # If the image type is None, we couldn't determine the image type
    if image_type is None:
        logger.warning("Could not determine the image type.")

    # Check the current extension
    base, ext = os.path.splitext(output_path)
    if image_type and ext[1:] != image_type:
        # If the extension is not correct, rename the file to include the correct extension
        correct_output_path = base + '.' + image_type
        os.rename(output_path, correct_output_path)
        return correct_output_path
    return output_path


def upscaler(upscaler_name: str) -> Optional[UpscalerData]:
    for upscaler_item in shared.sd_upscalers:
        if upscaler_item.name == upscaler_name:
            return upscaler_item
    return None


def face_restorer(face_restorer_name: str) -> Optional[FaceRestoration]:
    for face_restorer in shared.face_restorers:
        if face_restorer.name() == face_restorer_name:
            return face_restorer
    return None


def upscale_options(roop_options: RoopOptions) -> UpscaleOptions:
    roop_upscaler =  upscaler(roop_options.upscale_options.upscaler_name)
    roop_face_restorer = face_restorer(roop_options.upscale_options.face_restorer_name)
    return UpscaleOptions(
        scale=roop_options.upscale_options.scale,
        upscaler=roop_upscaler,
        face_restorer=roop_face_restorer,
        upscale_visibility=roop_options.upscale_options.upscale_visibility,
        restorer_visibility=roop_options.upscale_options.restorer_visibility,
    )


def get_private_path(request: Request, folder: str, filename: str) -> pathlib.Path:
    paths = Paths(gr.Request(request))
    private_dir: pathlib.Path = paths.private_outdir() / "roop" / folder
    if not private_dir.exists():
        private_dir.mkdir(parents=True, exist_ok=True)
    return private_dir / filename


def download_image_if_needed(request: Request, image_source: ImageSource, folder: str, filename: Optional[str] = None) -> str:
    if image_source.image_url is None and image_source.encoded_image is None and image_source.image_filepath is None:
        raise HTTPException(status_code=400, detail="image_url, encoded_image, image_filepath at least one is required")
    if image_source.image_filepath:
        if os.path.exists(image_source.image_filepath):
            return image_source.image_filepath
    if filename:
        image_name = filename
    else:
        image_name = str(uuid.uuid4())
    source_image_path = get_private_path(request, os.path.join("source", folder), image_name)
    if image_source.encoded_image:
        decoded_image = base64.b64decode(image_source.encoded_image)
        # Guess the extension from base64 encoded image string
        file_type = imghdr.what(None, decoded_image)
        if file_type is not None:
            source_image_path = f"{source_image_path}.{file_type}"
        with open(source_image_path, 'wb') as f:
            f.write(decoded_image)
        return str(source_image_path)
    if image_source.image_url:
        output_path = download_image(image_source.image_url, str(source_image_path))
        return output_path
    raise HTTPException(status_code=400, detail="Process image failed")


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64," + img_str.decode('utf-8')


def run_roop_on_single_image(
        request: Request, roop_options: RoopOptions) -> RoopResponse:
    start_time = datetime.now(timezone.utc)
    utc_date_str = start_time.strftime("%Y-%m-%d")
    task_id = roop_options.task_id
    face_image = download_image_if_needed(request, roop_options.source_image, f"face/{utc_date_str}", task_id)
    swap_image = download_image_if_needed(request, roop_options.swap_image, f"swap/{utc_date_str}", task_id)
    result: ImageResult = swap_face(
        Image.open(face_image),
        Image.open(swap_image),
        model=roop_options.model,
        faces_index={roop_options.face_index},
        upscale_options=upscale_options(roop_options)
    )
    result_image = result.image()
    if result_image is None:
        raise HTTPException(status_code=400, detail="Roop process image failed")
    output_filepath = get_private_path(request, f"output/{utc_date_str}", f"{task_id}.png")
    result_image.save(output_filepath)
    finish_time = datetime.now(timezone.utc)
    return RoopResponse(
        task_id=task_id,
        encoded_image=pil_image_to_base64(result_image),
        start_time=start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        finish_time=finish_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    )



def setup_apis(_: gr.Blocks, app: FastAPI):
    app.add_api_route(
        "/internal/roop/image",
        run_roop_on_single_image,
        methods=["POST"],
        response_model=RoopResponse)


script_callbacks.on_app_started(setup_apis)
