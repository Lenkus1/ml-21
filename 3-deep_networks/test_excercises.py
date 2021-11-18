from loguru import logger

import sys

logger.remove()
logger.configure(
    handlers=[
        {"sink": sys.stderr, "backtrace": False, "diagnose": True},
        {
            "sink": "../reports/excercises.log",
            "backtrace": False,
            "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        },
    ]
)


def shapetest(test: int, model):
    if test == 1:
        assert model._input_layers[0].input_shape[0][1] == 10
        logger.info("test 1 passed")

    if test == 2:
        try:
            assert len(model._input_layers) == 2
            assert (model._input_layers[0].input_shape[0][1] == 10) & (
                model._input_layers[1].input_shape[0][1] == 10
            )
            assert model.layers[1].output_shape[1] == 50
            assert model.layers[3].output_shape[1] == 60
            assert model.layers[4].output_shape[1] == 1
            logger.info("test 2 passed")
        except AssertionError:
            logger.exception("test 2 failed")
