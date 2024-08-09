import asyncio
import time
import bentoml

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import numpy as np
import tensorflow as tf

import os
import io
import json

settings = get_settings()


class MyService(Service):
    """
    Planet recognition service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="planet recognition service",
            slug="planet-recognition-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/planet-recognition",
        )
        self._logger = get_logger(settings)

        # Import the model to the model store from a local model folder
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", "celestial_bodies_classifier_model.bentomodel")
            bentoml.models.import_model(model_path)
        except bentoml.exceptions.BentoMLException:
            print("Model already exists in the model store - skipping import.")

        # Load model
        self._model = bentoml.keras.load_model("celestial_bodies_classifier_model")
        self._labels = os.path.join(os.path.dirname(__file__), "..", "labels.json")

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        raw = data["image"].data
        input_type = data["image"].type

        # Convert raw bytes to a PIL image
        image = Image.open(io.BytesIO(raw))

        # Resize the PIL image and convert it to grayscale ("L" mode)
        image = image.convert("L").resize((32, 32))

        # Convert the PIL image to a NumPy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Add a batch dimension and a channel dimension
        image_array = np.expand_dims(image_array, axis=0)  # Shape becomes (1, 32, 32)
        image_array = np.expand_dims(image_array, axis=-1)  # Shape becomes (1, 32, 32)

        # Create a TensorFlow dataset from the image array
        X_predict = tf.data.Dataset.from_tensor_slices(image_array).batch(1)

        # Use the model to predict images
        prediction = self._model.predict(X_predict)

        predicted_class = np.argmax(prediction, axis=-1)

        # Get label
        with open(self._labels, 'r') as file:
            class_labels = json.load(file)
        result = class_labels[int(predicted_class)]

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=str(result), type=FieldDescriptionType.APPLICATION_JSON)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """Planet recognition service
Service that recognizes pictures of planets.
"""
api_summary = """Planet recognition service
Service that recognizes pictures of planets.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Planet Recognition Service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
