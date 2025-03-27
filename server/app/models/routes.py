from flask import Blueprint

models_routes = Blueprint(
    "models",
    __name__,
)


@models_routes.route("/ft", methods=["GET"])
def finetuning():
    return "finetuning"
