from flask import Blueprint

home_routes = Blueprint(
    "home",
    __name__,
)


@home_routes.route("/", methods=["GET"])
def home():
    return "LLM Fine-Tuning Application"
