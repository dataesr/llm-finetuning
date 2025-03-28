from flask import Blueprint, jsonify
from app.home.requirements import get_installed_versions

home_routes = Blueprint(
    "home",
    __name__,
)


@home_routes.route("/", methods=["GET"])
def home():
    return "LLM Fine-Tuning Application"


@home_routes.route("/versions", methods=["GET"])
def versions():
    return jsonify(get_installed_versions())
